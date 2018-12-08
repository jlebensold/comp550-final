

## Experiments

```
python main.py perform_federated_training --with-replacement True --classes-per-worker 5 --same-initilization True
python main.py perform_federated_training --with-replacement True --classes-per-worker 5 --same-initilization False
python main.py perform_federated_training --with-replacement True --classes-per-worker 2 --same-initilization True
python main.py perform_federated_training --with-replacement True --classes-per-worker 2 --same-initilization False

python main.py perform_federated_training --with-replacement False --classes-per-worker 5 --same-initilization True
python main.py perform_federated_training --with-replacement False --classes-per-worker 5 --same-initilization False
python main.py perform_federated_training --with-replacement False --classes-per-worker 2 --same-initilization True
python main.py perform_federated_training --with-replacement False --classes-per-worker 2 --same-initilization False
```
