SHELL := /bin/bash
ROOT  := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
DATA  ?= $(ROOT)data

.PHONY: help setup doctor image-base image image-realsense openvins shim weights xfeat \
        euroc ori tumvi test clean

help:
	@echo "Setup, in order ('make setup' runs all of it):"
	@echo "  make image-base      CUDA / ROS 2 / Torch base image, built from source (long)"
	@echo "  make image           davio/dense-gpu:latest, the image everything runs in"
	@echo "  make openvins        build the pinned OpenVINS submodule inside the image"
	@echo "  make shim            build the pybind binding DAVIO drives OpenVINS through"
	@echo "  make weights         Depth Anything 3 (DA3-BASE) checkpoint"
	@echo "  make xfeat           XFeat + LighterGlue, for loop closure"
	@echo
	@echo "Data, under data/ (the only place the image can see):"
	@echo "  make euroc [SEQS='V1_01_easy MH_01_easy']"
	@echo "  make ori   [SEQ=r01]"
	@echo "  make tumvi"
	@echo
	@echo "  make test            the test suite, inside the image"
	@echo "  make image-realsense the image with the RealSense SDK, for a live camera"

setup: image-base image openvins shim weights xfeat
	./davio doctor

doctor:
	./davio doctor

image-base:
	docker build -f docker/Dockerfile.base -t davio/standalone-gpu:latest docker/

image:
	docker build -f docker/Dockerfile.dense -t davio/dense-gpu:latest docker/

image-realsense:
	docker build -f docker/Dockerfile.realsense -t davio/realsense:latest docker/

openvins:
	git submodule update --init thirdparty/open_vins thirdparty/Depth-Anything-3
	scripts/docker.sh bash -c 'cd thirdparty/open_vins && \
	  colcon build --packages-up-to ov_msckf --cmake-args -DCMAKE_BUILD_TYPE=Release'

shim:
	scripts/docker.sh native/build.sh

weights:
	scripts/fetch_da3_weights.sh base

xfeat:
	scripts/fetch_xfeat.sh

SEQS ?= V1_01_easy MH_01_easy
euroc:
	scripts/fetch_euroc.sh $(DATA) $(SEQS)

SEQ ?= r01
ori:
	python3 scripts/fetch_ori.py --data-root $(DATA) --sequences $(SEQ)
	scripts/docker.sh python3 scripts/convert_ori.py data/ori/$(SEQ)

tumvi:
	scripts/fetch_tumvi.sh $(DATA)

test:
	./davio test

# Build products only; data, runs and weights are left alone.
clean:
	rm -rf native/build src/openvins_ext*.so
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
