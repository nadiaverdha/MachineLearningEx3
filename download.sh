# download data files
mkdir cifar
curl -o cifar/cifar-10-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
gunzip -c cifar/cifar-10-python.tar.gz | tar xopf - -C cifar
