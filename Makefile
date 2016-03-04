tests:
	python fntools/fntools.py

release:
	rm -r dist
	python setup.py sdist bdist_wheel
	twine upload dist/*
