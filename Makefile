tests:
	python fntools/fntools.py

release:
	rm -r dist
	python setup.py sdist
	twine upload dist/*
