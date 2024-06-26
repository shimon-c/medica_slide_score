## DevOps tools :

**If you dont have Python 3.11 installed on your host.**

```bash
sudo ./do/install_python3.11.sh
```

**Create Python virtual env to be used from your IDE (Pycharm ...)**

A venv directory will be create and it will include all the requirements from
the pyproject.toml

```bash
./do/venv.sh
```

To create a runtime venv for evaluate only.

```bash
./do/venv.sh --install development
```