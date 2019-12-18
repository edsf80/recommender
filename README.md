# Recommender
Recommender for test cases. Steps do make it work.

Requires python 3.

## Application dependencies

1. Install pandas

```
$ pip3 install pandas
```

2. Install scikit learn

```
$ pip3 install scikit-learn
```

## Running application

The file main.py has the main function which calls the recommendation method. The method getRecommendations requires two parameters, the first one is the dictionary representing a new User Story to be registered that must have the following attributes:

- Módulo: The classfication User Story module. The possible values are: Cadastro, Autenticação, and Management.

```
$ python3 main.py
```