# One-time setup. Day-to-day use is ./davio.
SHELL := /bin/bash
DATA  ?= data

.PHONY: help setup image openvins binding weights xfeat euroc ori test clean

help:
	@echo "make setup        everything below, in order (about an hour)"
	@echo "make image        the container image everything runs in (CUDA, ROS 2, Torch)"
	@echo "make openvins     build the OpenVINS submodule inside the image"
	@echo "make binding      build the Python binding of OpenVINS"
	@echo "make weights      Depth Anything 3 checkpoint [MODEL=base|large|giant]"
	@echo "make xfeat        XFeat and LighterGlue, for matching"
	@echo "make euroc        EuRoC sequences into data/euroc [SEQS='V1_01_easy V1_02_medium']"
	@echo "make ori          an ORI sequence into data/ori [SEQ=r01]"
	@echo "make test         the test suite, inside the image"

setup: image openvins binding weights xfeat

image:
	docker build -f docker/Dockerfile.base -t davio/standalone-gpu:latest docker/
	docker build -f docker/Dockerfile.dense -t davio/dense-gpu:latest docker/

openvins:
	git submodule update --init thirdparty/open_vins thirdparty/Depth-Anything-3
	scripts/docker.sh bash -c 'cd thirdparty/open_vins && colcon build --packages-up-to ov_msckf --cmake-args -DCMAKE_BUILD_TYPE=Release'

binding:
	scripts/docker.sh native/build.sh

MODEL ?= base
weights:
	scripts/fetch_da3_weights.sh $(MODEL)

xfeat:
	scripts/fetch_xfeat.sh

SEQS ?= V1_01_easy V1_02_medium
euroc:
	scripts/fetch_euroc.sh $(DATA) $(SEQS)

SEQ ?= r01
ori:
	python3 scripts/fetch_ori.py --data-root $(DATA) --sequences $(SEQ)
	scripts/docker.sh python3 scripts/convert_ori.py data/ori/$(SEQ)
	scripts/docker.sh python3 scripts/ori_reference.py data/ori/$(SEQ)

test:
	./davio test

clean:
	rm -rf native/build src/openvins_ext*.so
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
