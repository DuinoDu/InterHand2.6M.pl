gpus=0,
batch_size=8

all:

train:
	python scripts/train.py --gpus 0,1,2,3 --batch_size $(batch_size)

debug:
	python scripts/train.py --gpus 0, --batch_size 1

build:
	python setup.py build

upload:
	python setup.py bdist_wheel upload -r hobot-local

clean:
	@rm -rf build dist src/*.egg-info

test:
	pytest --capture=no

pep8:
	autopep8 src/interhand --recursive -i

lint:
	pylint src/interhand --reports=n

lintfull:
	pylint src/interhand

install:
	python setup.py install

uninstall:
	python setup.py install --record install.log
	cat install.log | xargs rm -rf 
	@rm install.log
