
build:
	python setup.py build

upload:
	python setup.py bdist_wheel upload -r hobot-local

clean:
	@rm -rf build dist src/*.egg-info

test:
	python /usr/bin/nosetests -s tests --verbosity=2 --rednose --nologcapture

pep8:
	autopep8 src/mnist_pl --recursive -i

lint:
	pylint src/mnist_pl --reports=n

lintfull:
	pylint src/mnist_pl

install:
	python setup.py install

uninstall:
	python setup.py install --record install.log
	cat install.log | xargs rm -rf 
	@rm install.log
