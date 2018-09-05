.PHONY: test

help:
	@echo "  env            create a development environment using virtualenv"
	@echo "  clean          remove unwanted stuff"

env:
	pip3 install virtualenv && \
	virtualenv --always-copy -p `which python3` .env && \
	. .env/bin/activate && \
	make deps && \
	pre-commit install

deps:
	pip3 install  --upgrade -r requirements.txt

clean:
	find . -name '*.pyc' -exec rm -f {} \; && \
  	rm -rf temp && \
  	rm -rf dist && \
  	rm -fr .env
