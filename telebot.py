import telebot
import random
from telebot import types
import pandas as pd
import numpy as np
import time
import os

lamour = ['Маменька красавица!', 'Совсем меня разлюбила(', 'Маменька у меня принцесса', 'Пора отдыхать',
          'Люблю, люблю маменьку!']
lamour = np.array(lamour)

def Sv(file1, file2):
    df_1C = pd.read_excel(file1, dtype='string', names=['code', 'quantity'])
    df_Sveta = pd.read_excel(file2,dtype='string', names=['code', 'quantity'])
    df_1C['quantity'] = df_1C['quantity'].str.replace(',', '.')
    df_Sveta['quantity'] = df_Sveta['quantity'].str.replace(',', '.')
    df_1C['quantity'] = df_1C['quantity'].str.replace(' ', '')
    df_Sveta['quantity'] = df_Sveta['quantity'].str.replace(' ', '')
    df_1C['quantity'] = df_1C['quantity'].astype('float')
    df_Sveta['quantity'] = df_Sveta['quantity'].astype('float')
    df = df_1C.merge(df_Sveta, on='code', how='outer')
    x = df[df['quantity_x'] != df['quantity_y']]
    x.to_excel('Сличительная.xlsx', index=False)

bot = telebot.TeleBot('5407511525:AAG5Z9x6beOi5c49aeuH9DEspjdrPmQcEhw')


@bot.message_handler(commands=["start"])
def start(m, res=False):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    item1 = types.KeyboardButton("Любовь")
    item2 = types.KeyboardButton("Сводная таблица")
    item3 = types.KeyboardButton("Начать работу")
    markup.add(item1)
    markup.add(item2)
    markup.add(item3)
    bot.send_message(m.chat.id, 'Привет, маменька!', reply_markup=markup)


@bot.message_handler(content_types=["text"])
def handle_text(message):
    if message.text.strip() == 'Любовь':
        answer = np.random.choice(lamour)
    if message.text.strip() == 'Сводная таблица':
        answer = 'Сбрось сначала файл Светы, потом свой (Назови первый "1.xlsx", второй "2.xlsx")'
    if message.text.strip() == 'Начать работу':
        Sv('1.xlsx', '2.xlsx')
        time.sleep(10)
        doc = open('Сличительная.xlsx', 'rb')
        bot.send_document(message.chat.id, doc)
        os.remove('/Users/akonkina/pythonProject/Сличительная.xlsx')
        answer = 'Готово, целую!'

    bot.send_message(message.chat.id, answer)


@bot.message_handler(content_types=["document"])
def handle_docs_photo(message):
    try:
        chat_id = message.chat.id

        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        src = '/Users/akonkina/pythonProject/' + message.document.file_name
        print(src)
        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)
        bot.reply_to(message, "Получил")
        bot.send_message(message.chat.id, 'Когда отправишь два файла, нажми "Начать работу"')
    except Exception as e:
        bot.reply_to(message, e)


bot.polling(none_stop=True, interval=0)
