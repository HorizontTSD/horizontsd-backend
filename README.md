<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <img src="/readme_materials/logo_line_white.svg" width="80%" alt="DeepSeek-V3" />
</div>
<hr>

[English description](#english-description) | [Русское описание](#русское-описание)

## English description
Welcome to the project!

## Русское описание
Добро пожаловать в проект!



<p align="center">

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
![Code Coverage](coverage.svg)

</p>

# Запуск на своей машине

#### Установка зависимостей
```bash
pip install pdm
pdm install
```

Активация окружения
```bash
source .venv/bin/activate
```

#### Пример добавления/удаления новых зависимостей
```bash
pdm add pandas
pdm remove pandas
```

Запуск на своей машине
```bash
python -m src.server
```

После запуска ui микросервис адоступен по адресу
```bash
http://0.0.0.0:7070/template_fast_api/v1/#/
```


# Запуск контейнера публично

### Строим контейнер
```bash
sudo docker build -t horizontsd_backend .
```
Узнаем его IMAGE ID 
```bash
sudo docker images
```

```bash
sudo docker run -d -p 7071:7071 <IMAGE ID>
```

```bash
sudo docker run -d -p 80:7071 <IMAGE ID>
```

```bash
sudo docker run -d -p 7071:80 <IMAGE ID>
```



# Запуск контейнера локально

### Строим контейнер
```bash
sudo docker build -t horizontsd_backend .
```
Узнаем его ID
```bash
sudo docker images
```

```bash
sudo docker run -p 7071:7071 <IMAGE ID>
```