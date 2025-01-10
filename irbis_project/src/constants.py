from pathlib import Path

PATH_TO_NOOBF_DATA = Path(__file__).parent.parent / 'data/result/no_obfuscation'
FIRST_STAGE_UNIFICATION_MAPPER = {
    "Бурый медведь": "Медведь",
    "Гималайский медведь": "Медведь",
    "Медведь": "Медведь",
    "Кабан": "Кабан",
    "Изюбрь": "Оленевые",
    "Пятнистый олень": "Оленевые",
    "Марал": "Оленевые",
    "Сибирская косуля": "Оленевые",
    "Косуля": "Оленевые",
    "Азиатский барсук": "Куньи",
    "Росомаха": "Куньи",
    "Соболь": "Куньи",
    "Харза": "Куньи",
    "Амурский лесной кот": "Кошки",
    "Манул": "Кошки",
    "Рысь": "Кошки",
    "Тигр": "Пантеры",
    "Ирбис": "Пантеры",
    "Аргали": "Полорогие",
    "Козерог": "Полорогие",
    "Волк": "Собачие",
    "Лиса": "Собачие",
    "Енотовидная собака": "Собачие",
    "Заяц": "Заяц",
    "Кабарга": "Кабарга",
    "Сурок": "Сурок"
}

FIRST_STAGE_INDEX_MAPPER = {cls_name: idx for idx, cls_name in enumerate(['Заяц', 'Кабан', 'Кошки', 'Куньи', 'Медведь', 'Оленевые', 'Пантеры', 'Полорогие', 'Собачие', 'Сурок'])}

SIZE_IMAGES_RESIZED = (1920, 1080)
SIZE_CROP = (0, 30, 1920, 1020)
