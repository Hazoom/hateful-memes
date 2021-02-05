.PHONY: all
all: black_check

export ROOT_DIR := $(realpath .)

.PHONY: black
black:
	black $(ROOT_DIR)/src --config=./pyproject.toml

.PHONY: black_check
black_check:
	black --check $(ROOT_DIR)/src --config=./pyproject.toml
