"""Transport-independent online estimator. Only one owner thread calls step()."""
from collections import deque, OrderedDict
import json,shutil
from pathlib import Path
import time
import numpy as np
from ..backends.openvins import OpenVinsBackend
from ..data.images import build_rectifier, PassThrough
from ..data.types import CameraModel
from ..init.jpl import quat_to_rot
from .geometry import json_safe, camera_pose, tum_line
from .configuration import materialize, read_relative, read_yaml
from .workers import Worker, feed_packet, vision_loop, shadow_loop

def ramp_correction(start,target,fraction):
    """Slerp/lerp between two odometry->map corrections; fraction in [0, 1]."""
    from scipy.spatial.transform import Rotation,Slerp
    f=float(np.clip(fraction,0.,1.))
    out=np.eye(4)
    out[:3,:3]=Slerp([0.,1.],Rotation.from_matrix(np.array([start[:3,:3],target[:3,:3]])))(f).as_matrix()
    out[:3,3]=(1.-f)*start[:3,3]+f*target[:3,3]
    return out


class Engine:
    def __init__(self,cfg,rig,out,on_pose=None,meta=None):
        self.cfg,self.out,self.on_pose=cfg,Path(out).resolve(),on_pose
        self.meta=dict(meta or {})
        self.out.mkdir(parents=True,exist_ok=False)
        self.rig=Path(rig).resolve()
        self.selected=None;self.native=None;self.shadow=None;self.vision=None
        self.attempted=False;self.vision_failed=False;self.count=0;self.dropped=0;self.submitted=0
        self.attempts=0;self.candidate_started=-np.inf
        self.start=None;self.last_t=-np.inf;self.last_pose=-np.inf;self.last_sample=-np.inf;self.last_attempt=-np.inf
        self.opened=time.monotonic();self.first_publication=None;self.ages=[]
        self.events=(self.out/'events.jsonl').open('w',buffering=1)
        self.trajectory=(self.out/'trajectory.tum').open('w',buffering=1)
        self.camera_trajectory=(self.out/'camera_trajectory.tum').open('w',buffering=1)
        # Odometry frame -> corrected map frame. Identity until the back-end reports
        # one, and always the latest available: the map lags, the odometry does not.
        self.map_correction=np.eye(4)
        # A correction that jumps further than mapping.correction_jump_* from the active one
        # is slid in over correction_ramp_s of sensor time so the live stream has no jumps;
        # the loop-closed keyframe trajectory (export_trajectory.py) carries the full step.
        self.correction_from=np.eye(4);self.correction_start=None
        # Which back-end commit the current correction came from, so every published pose
        # can be tied to the map version it was corrected with (frame consistency).
        self.map_version=None;self.map_activation=None
        self.map_trajectory=(self.out/'map_trajectory.tum').open('w',buffering=1)
        self.history=deque(maxlen=cfg['assistance']['history_frames'])
        self.imu=deque();self.frames=OrderedDict();self.assist_ids=deque(maxlen=5)
        self.map_frames=deque(maxlen=cfg['mapping']['window_frames']);self.last_map=-np.inf
        self.last_diagnostics=-np.inf
        self.status='starting';self.error=None;self.closing=False
        self.write_status()
        try:
            # Calibration mode decides what the estimator is told (configuration.materialize)
            # and what the vision worker may use. In free modes no native instance exists:
            # there is nothing to construct it from.
            mode=cfg.setdefault('calibration',{}).get('mode','supplied')
            chain=read_relative(self.rig,'relative_config_imucam')['cam0']
            resolution=tuple(int(x) for x in chain['resolution'])
            gravity=float(read_yaml(self.rig).get('gravity_mag',9.81))
            self.rectifier_model=None
            if mode=='free':
                self.rectifier=PassThrough(tone='shift8')
                runtime=dict(mode=mode,R_CtoI=None,p_CinI=None,K=None,D=None,resolution=resolution,gravity_mag=gravity)
            else:
                fx,fy,cx,cy=chain['intrinsics']
                camera=CameraModel(K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]]),
                    D=np.array(chain['distortion_coeffs']),model=chain['distortion_model'],resolution=resolution)
                self.rectifier=build_rectifier(camera,fov_deg=90.,out_size=512,tone='shift8')
                T=np.asarray(chain['T_imu_cam'],float)
                runtime=dict(mode=mode,R_CtoI=T[:3,:3].tolist() if mode=='supplied' else None,
                    p_CinI=T[:3,3].tolist() if mode=='supplied' else None,
                    K=camera.K.tolist(),D=camera.D.tolist(),resolution=resolution,gravity_mag=gravity)
            cfg['calibration_runtime']=runtime      # read by the vision worker; recorded in run.json
            if mode!='supplied':
                if not cfg['assistance']['enabled']:raise ValueError('free calibration modes need assistance')
                cfg['assistance']['native_priority']=False
            from .assistance import CalibrationAssistant
            ok,reason=CalibrationAssistant.shadow_schedule_ok(cfg['assistance'])
            if not ok:raise ValueError('Shadow calibration cannot validate: '+reason)
            if mode=='supplied':
                native_path=materialize(self.rig,self.out/'native_config',cfg['openvins']['overrides'])
                self.native=OpenVinsBackend(native_path)
            import cv2
            cv2.setRNGSeed(int(cfg['seed']))
            if cfg['assistance']['enabled'] or cfg['mapping']['enabled']:
                self.vision=Worker(vision_loop,(cfg,str(self.out)),capacity=1)
            self.status='running';self.write_status()
        except BaseException as exc:
            self.close(error=str(exc));raise

    def event(self,kind,**kw):
        self.events.write(json.dumps(json_safe(dict(kind=kind,wall=time.monotonic(),**kw)),allow_nan=False)+'\n')

    def write_status(self):
        value=dict(self.meta)
        value.update(schema_version=3,state=self.status,error=self.error,selected=self.selected,
            n_states=self.count,first_sensor_time=self.start,last_sensor_time=None if not np.isfinite(self.last_t) else self.last_t,
            first_publication_wall_s=self.first_publication,elapsed_wall_s=time.monotonic()-self.opened,
            config=self.cfg,dropped_optional_tasks=self.dropped,submitted_optional_tasks=self.submitted,
            vision_failed=self.vision_failed,ground_truth_used=False,
            integrated_mapping_success=(not self.cfg['mapping']['enabled'] or
                (not self.vision_failed and (self.out/'map/map_index.json').exists())),
            publication_age_p50_s=float(np.median(self.ages)) if self.ages else None,
            publication_age_p95_s=float(np.percentile(self.ages,95)) if self.ages else None)
        temp=self.out/'run.tmp.json';temp.write_text(json.dumps(json_safe(value),indent=2,allow_nan=False));temp.replace(self.out/'run.json')

    def publish(self,state):
        t=float(state['t'])
        if t<=self.last_pose:return
        self.last_pose=t;self.count+=1
        if self.first_publication is None:self.first_publication=time.monotonic()-self.opened
        body=np.eye(4);body[:3,:3]=quat_to_rot(np.asarray(state['q_GtoI'])).T;body[:3,3]=state['p']
        cam=camera_pose(state)
        self.trajectory.write(tum_line(t,body));self.camera_trajectory.write(tum_line(t,cam))
        active,fraction=self.active_correction(t)
        self.map_trajectory.write(tum_line(t,active@body))
        item=self.frames.get(round(t*1e9))
        age=None if item is None else time.monotonic()-item['arrival_wall']
        if age is not None:self.ages.append(age)
        self.event('pose',t=t,selected=self.selected,receipt_to_publish_s=age,state=state,
            correction_version=self.map_version,correction_fraction=fraction)
        self.log_calibration(t)
        if self.cfg['calibration'].get('mode')=='free' and 'cam_k' in state and self.cfg['mapping']['enabled']:
            self.refresh_rectifier(state)
        if self.on_pose:self.on_pose(t,body,state)
        if self.closing or not self.cfg['mapping']['enabled'] or item is None or self.vision is None or self.vision_failed:return
        if t-self.last_map<self.cfg['mapping']['keyframe_period_s']:return
        self.last_map=t;self.map_frames.append((round(t*1e9),item['neural'],cam,item.get('color')))
        if len(self.map_frames)<self.map_frames.maxlen:return
        ids,images,poses,colors=zip(*self.map_frames)
        task=dict(kind='map',stamps=list(ids),times=[s*1e-9 for s in ids],images=list(images),
            colors=list(colors) if all(c is not None for c in colors) else None,
            K=self.rectifier.K_new,camera_poses=np.asarray(poses),available_sensor_time=self.last_t,
            interval=(self.meta.get('interval_start'),self.meta.get('interval_end')) if self.meta.get('interval_end') is not None else None,
            # Age is charged from the newest contributing frame's receipt, not from
            # submission: queueing delay is part of how stale the map is.
            arrival_wall=item['arrival_wall'],submitted_wall=time.monotonic())
        if self.vision.submit(task):
            self.submitted+=1
            overlap=int(self.cfg['mapping'].get('overlap_frames', 2))
            while len(self.map_frames)>overlap:self.map_frames.popleft()
        else:self.dropped+=1;self.event('optional_drop',task='map',t=t)

    def active_correction(self,t):
        """The odometry->map correction in force at sensor time t and its ramp fraction."""
        if self.correction_start is None:return self.map_correction,1.
        f=(t-self.correction_start)/max(float(self.cfg['mapping'].get('correction_ramp_s',0.)),1e-9)
        if f>=1.:self.correction_start=None;return self.map_correction,1.
        return ramp_correction(self.correction_from,self.map_correction,f),float(max(f,0.))

    def refresh_rectifier(self,state):
        k=np.asarray(state['cam_k'],float)
        if k.shape[0]<8 or not np.isfinite(k).all() or k[0]<=0 or k[1]<=0:return
        K=np.array([[k[0],0,k[2]],[0,k[1],k[3]],[0,0,1.]]);D=k[4:8]
        res=tuple(self.cfg['calibration_runtime']['resolution'])
        old=self.rectifier_model
        if old is not None:
            w,h=res;corners=np.array([[0,0,1],[w,0,1],[0,h,1],[w,h,1.]],float)
            moved=np.abs(corners@np.linalg.inv(old.K).T-corners@np.linalg.inv(K).T)[:,:2].max()*max(k[0],k[1])
            if moved<float(self.cfg['calibration']['rectifier_refresh_px']) and np.abs(D-old.D).max()<1e-3:return
        self.rectifier_model=CameraModel(K=K,D=D,model='radtan',resolution=res)
        self.rectifier=build_rectifier(self.rectifier_model,fov_deg=90.,out_size=512,tone='shift8')
        self.map_frames.clear()
        self.event('rectifier_refreshed',t=float(state['t']),cam_k=k.tolist())

    def log_calibration(self,t):
        period=self.cfg.get('diagnostics_period_s',0.)
        if not period or t-self.last_diagnostics<period:return
        self.last_diagnostics=t
        backend=self.native if self.selected=='native' else None
        if backend is None:
            self.event('calibration',t=t,available=False,
                       reason='diagnostics are only wired for a selected native instance')
            return
        diagnostics=backend.diagnostics()
        if diagnostics is None:
            self.event('calibration',t=t,available=False,reason='shim reported none')
            return
        record=dict(t=t,available=True,n_clones=diagnostics.get('n_clones'),
                    filter_time=diagnostics.get('t'))
        imu=np.asarray(diagnostics.get('cov_imu'),float) if 'cov_imu' in diagnostics else None
        if imu is not None and imu.shape==(15,15):
            sigma=np.sqrt(np.clip(np.diag(imu),0,None))
            # [theta p v bg ba], matching the binding's documented ordering.
            record.update(sigma_theta_deg=np.degrees(sigma[0:3]).tolist(),
                          sigma_p_m=sigma[3:6].tolist(),sigma_v_ms=sigma[6:9].tolist(),
                          sigma_bg_rads=sigma[9:12].tolist(),sigma_ba_ms2=sigma[12:15].tolist())
        calib=np.asarray(diagnostics.get('cov_calib0'),float) if 'cov_calib0' in diagnostics else None
        if calib is not None and calib.shape==(6,6):
            sigma=np.sqrt(np.clip(np.diag(calib),0,None))
            eigenvalues=np.linalg.eigvalsh((calib+calib.T)/2.)
            positive=eigenvalues[eigenvalues>0]
            record.update(
                calib_sigma_rot_deg=np.degrees(sigma[0:3]).tolist(),
                calib_sigma_trans_m=sigma[3:6].tolist(),
                # Information is the inverse of covariance; its spread over the six
                # directions is what distinguishes an excited axis from a frozen one.
                calib_information_eigenvalues=(1./positive[::-1]).tolist() if positive.size else [],
                calib_condition=float(eigenvalues[-1]/eigenvalues[0])
                if eigenvalues[0]>0 else None)
        self.event('calibration',**record)

    def step(self,packet):
        """Packet: camera t/image, preceding+bracketing IMU, camera receipt wall time."""
        t=float(packet['t'])
        if t<=self.last_t:raise ValueError('Camera times must increase')
        if self.start is None:self.start=t
        self.last_t=t
        if not packet['imu'] or packet['imu'][-1][0]<t:raise ValueError('IMU must bracket the camera')
        self.imu.extend(packet['imu'])
        while self.imu and self.imu[0][0]<t-self.cfg['assistance']['imu_history_s']:self.imu.popleft()
        neural=self.rectifier([packet['image']])[0]
        item=dict(neural=neural,arrival_wall=packet['arrival_wall'])
        # Colour is rectified through the SAME maps as the grey frame, so it lands on the
        # identical grid, and is kept only while mapping is on: it is four times the memory
        # of the grey frame and nothing but the map reads it.
        if packet.get('color') is not None and self.cfg['mapping']['enabled']:
            item['color']=self.rectifier.color([packet['color']])[0]
        self.frames[round(t*1e9)]=item
        while len(self.frames)>self.cfg['assistance']['history_frames']:self.frames.popitem(last=False)
        self.history.append(packet)
        state=feed_packet(self.native,packet) if self.native is not None else None
        # native_priority=false is the assisted-only ablation arm: the native instance
        # still runs (and still owns the run if assistance is off) but cannot win.
        native_selectable=self.cfg['assistance']['native_priority'] or not self.cfg['assistance']['enabled']
        if self.selected is None and state is not None and native_selectable:
            self.selected='native';self.event('selected',estimator='native',t=t)
            if self.shadow:self.shadow.close();self.shadow=None
        if self.selected=='native' and state is not None:self.publish(state)
        if self.shadow:
            submitted=self.shadow.submit(packet)
            if not submitted and self.selected=='assisted':
                # Dataset replay may deliver a burst after native initialization stalls.
                # Drain output while waiting for input space to avoid a two-queue deadlock.
                deadline=time.monotonic()+self.cfg.get('selected_queue_timeout_s', 2.)
                while not submitted and time.monotonic()<deadline:
                    for result in self.shadow.drain():
                        if result['kind']=='error':raise RuntimeError(result['error'])
                        if result.get('state') is not None:self.publish(result['state'])
                    if not self.shadow.process.is_alive():raise RuntimeError('Selected estimator died')
                    submitted=self.shadow.submit(packet)
                    if not submitted:time.sleep(.001)
                if not submitted:raise RuntimeError('Selected estimator ingress timeout')
            if not submitted:
                # Same retry rule as the warm-up timeout below: a rejected candidate must not
                # end startup assistance (it did until 2026-09-13: `attempted` stayed True).
                self.event('candidate_rejected',reason='cannot catch up',t=t,attempt=self.attempts);self.shadow.close();self.shadow=None
                self.attempted=self.attempts>=int(self.cfg['assistance'].get('max_attempts',1))
            else:
                for result in self.shadow.drain():
                    if result['kind']=='error':
                        if self.selected=='assisted':raise RuntimeError(result['error'])
                        self.event('candidate_rejected',reason=result['error'],t=t,attempt=self.attempts);self.shadow.close();self.shadow=None
                        self.attempted=self.attempts>=int(self.cfg['assistance'].get('max_attempts',1));break
                    state=result['state']
                    if self.selected is None and state is not None and result['warm_frames']>=self.cfg['assistance']['warm_frames']:
                        if 0<=t-float(state['t'])<=self.cfg['assistance']['max_catchup_lag_s']:
                            self.selected='assisted';self.event('selected',estimator='assisted',t=t)
                            if self.native:self.native.close()
                            self.native=None
                    if self.selected=='assisted' and state is not None:self.publish(state)
        self.poll()
        # Startup assistance proposes BEFORE selection and stops. Shadow calibration keeps
        # the same solver running afterwards purely to record what it would have said; the
        # result is logged and never reaches the filter. It shares the capacity-one vision
        # queue with mapping, so a shadow attempt can cost a map window, which stays
        # visible in dropped_optional_tasks.
        startup=(self.selected is None and not self.attempted
                 and t-self.start<=self.cfg['assistance']['budget_s'])
        shadow=(self.selected is not None
                and bool(self.cfg['assistance'].get('shadow_continuous',False)))
        if (startup or shadow) and self.vision and not self.vision_failed and self.cfg['assistance']['enabled']:
            attempt_period=(self.cfg['assistance']['attempt_period_s'] if startup
                            else float(self.cfg['assistance'].get('shadow_period_s',5.)))
            if t-self.last_sample>=self.cfg['assistance']['sample_period_s']:
                self.last_sample=t;self.assist_ids.append(round(t*1e9))
                if len(self.assist_ids)==5 and t-self.last_attempt>=attempt_period:
                    task=dict(kind='assist',times=[s*1e-9 for s in self.assist_ids],
                        images=[self.frames[s]['neural'] for s in self.assist_ids],K=self.rectifier.K_new,
                        imu=list(self.imu),available_sensor_time=t,
                        arrival_wall=packet['arrival_wall'],submitted_wall=time.monotonic())
                    if self.vision.submit(task):self.submitted+=1;self.last_attempt=t
                    else:self.dropped+=1
        # A candidate that never warms up is abandoned so a later window can try again,
        # up to max_attempts; the init study counts every attempt and every failure.
        if (self.shadow and self.selected is None and
                t-self.candidate_started>float(self.cfg['assistance'].get('candidate_timeout_s',3.))):
            self.event('candidate_rejected',reason='no warm state within candidate_timeout_s',t=t,attempt=self.attempts)
            self.shadow.close();self.shadow=None
            self.attempted=self.attempts>=int(self.cfg['assistance'].get('max_attempts',1))
        if self.selected and t-self.last_pose>self.cfg['max_tracking_gap_s']:raise RuntimeError('Tracking gap; no automatic gauge reset')
        if self.shadow and not self.shadow.process.is_alive():
            if self.selected=='assisted':raise RuntimeError('Selected estimator died')
            # A hard crash (no error result) must be visible and must not end the attempts.
            self.event('candidate_rejected',reason='estimator process died without an error result',t=t,attempt=self.attempts)
            self.shadow.close();self.shadow=None
            self.attempted=self.attempts>=int(self.cfg['assistance'].get('max_attempts',1))
        if self.count%20==0:self.write_status()

    def poll(self):
        if not self.vision:return
        for item in self.vision.drain():
            self.event('vision',result=item)
            if item['kind']=='error':self.vision_failed=True;continue
            result=item['payload']
            if item['kind']=='assist' and self.selected:
                # A shadow epoch: what the calibration solver would have proposed at this
                # point in the run. Recorded as evidence, never applied.
                self.event('shadow_calibration',t=self.last_t,
                           sensor_time=item.get('sensor_time'),payload=result,
                           applied=False,
                           note='estimated after estimator selection; no feedback path exists')
            if item['kind']=='map' and result.get('T_map_odom') is not None:
                new=np.asarray(result['T_map_odom'],float)
                active,_f=self.active_correction(self.last_t)
                delta=np.linalg.inv(active)@new
                jump_m=float(np.linalg.norm(delta[:3,3]))
                jump_deg=float(np.degrees(np.arccos(np.clip((np.trace(delta[:3,:3])-1.)/2.,-1.,1.))))
                m=self.cfg['mapping']
                ramped=(m.get('correction_ramp_s',0.)>0 and
                    (jump_m>float(m.get('correction_jump_m',np.inf)) or jump_deg>float(m.get('correction_jump_deg',np.inf))))
                self.correction_from,self.correction_start=(active,self.last_t) if ramped else (new,None)
                self.map_correction=new
                self.map_version=result.get('submap');self.map_activation=dict(sensor_time=self.last_t,wall=time.monotonic())
                self.event('correction_activated',t=self.last_t,version=self.map_version,
                    T_map_odom=self.map_correction.tolist(),completed_wall=item.get('completed_wall'),
                    jump_m=jump_m,jump_deg=jump_deg,ramped=ramped)
            if item['kind']!='assist' or result['status']!='released' or self.selected or self.attempted:continue
            age=self.last_t-result['available_sensor_time']
            if age>self.cfg['assistance']['max_candidate_age_s']:
                # A released candidate that took too long to compute is dropped, visibly.
                self.event('candidate_stale',t=self.last_t,age_s=age,bootstrap=result.get('bootstrap') is not None);continue
            if self.last_t-self.start>self.cfg['assistance']['budget_s']:
                self.event('candidate_stale',t=self.last_t,age_s=age,reason='past budget_s');continue
            self.attempted=True
            try:
                # A rejected candidate leaves its config behind; the next attempt replaces it
                # (materialize refuses to overwrite, which silently ended every retry before).
                shutil.rmtree(self.out/'assisted_config',ignore_errors=True)
                path=materialize(self.rig,self.out/'assisted_config',self.cfg['openvins']['overrides'],result,
                    mode=self.cfg['calibration'].get('mode','supplied'),priors=self.cfg['calibration'].get('priors'))
                (self.out/'candidate.json').write_text(json.dumps(result,indent=2,allow_nan=False))
                self.shadow=Worker(shadow_loop,(str(path),self.candidate_history(result.get('bootstrap')),
                    self.cfg['seed'],result.get('bootstrap')),capacity=self.cfg['assistance']['shadow_queue_frames'])
                self.attempts+=1;self.candidate_started=self.last_t
                self.event('candidate_started',t=self.last_t,attempt=self.attempts,
                    bootstrap=result.get('bootstrap') is not None)
            except (OSError,ValueError,RuntimeError) as exc:self.event('candidate_rejected',reason=str(exc))
        if not self.vision.process.is_alive():self.vision_failed=True

    def candidate_history(self,bootstrap):
        history=list(self.history)
        preroll=self.cfg['assistance'].get('candidate_preroll_s')
        if preroll is None or not bootstrap:return history
        start=float(bootstrap['t'])-float(preroll)
        trimmed=[p for p in history if float(p['t'])>=start]
        return trimmed if trimmed else history[-1:]

    def close(self,error=None):
        if self.status in ('completed','failed'):return
        self.closing=True
        # Grace only for already-submitted work; never enqueue extra mapping at shutdown.
        if self.vision or self.shadow:
            until=time.monotonic()+self.cfg['shutdown_drain_s']
            while time.monotonic()<until:
                if self.selected=='assisted' and self.shadow:
                    for result in self.shadow.drain():
                        if result['kind']=='error':error=error or result['error']
                        elif result.get('state') is not None:self.publish(result['state'])
                for item in (self.vision.drain() if self.vision else []):
                    self.event('vision',result=item)
                    if item['kind']=='error':self.vision_failed=True
                time.sleep(.02)
        for worker in (self.shadow,self.vision):
            if worker:worker.close()
        if self.native:self.native.close()
        self.status='failed' if error or not self.count else 'completed';self.error=error or (None if self.count else 'no initialization')
        self.write_status()
        for stream in (self.events,self.trajectory,self.camera_trajectory,self.map_trajectory):
            stream.close()
