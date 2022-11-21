First, load Python/3.10.4-GCCcore-11.3.0 module.
Then create python virtual environment by 'python3 -m venv py-test-env'.
Afterwards activate python virtual environmenr by 'source py-test-env/bin/activate'.
Install h5py module by 'python3 -m pip install h5py'
Finally, execute script 'run_tests.sh' with one parameter being the directory containing 'nbody' binary.
To deactivate environment execute 'deactivate' 
Caution! The script is intended to be executed only from the 'tests' directory.

