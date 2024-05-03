import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI
import locale
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties
from matplotlib.colors import LinearSegmentedColormap

locale.setlocale(locale.LC_ALL, 'ru_RU.UTF-8')

def load_data(uploaded_file):
    try:
        data = pd.read_csv(uploaded_file)
        data['Дата'] = pd.to_datetime(data['Дата'], errors='coerce')  # Extract date part
        data.set_index('Дата', inplace=True)
        data['Сумма'] = pd.to_numeric(data['Сумма'], errors='coerce')
        return data
    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")
        return None



def adjust_cmap(cmap_name, min_val=0.2, max_val=1.0):
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(min_val, max_val, 256))
    new_cmap = LinearSegmentedColormap.from_list("trimmed_" + cmap_name, colors)
    return new_cmap

def plot_financial_data(data):
    income_data = data[data['Доходы/Расходы'] == 'Доходы']['Сумма'].resample('D').sum()
    expense_data = data[data['Доходы/Расходы'] == 'Расходы']['Сумма'].resample('D').sum()

    fig, ax = plt.subplots(figsize=(15, 6))

    income_data.plot(kind='bar', ax=ax, color='#213a85', label='Доходы', width=0.4, position=0)
    expense_data.plot(kind='bar', ax=ax, color='#e43b29', label='Расходы', width=0.4, position=1)

    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))

    plt.title('Динамика доходов и расходов по дням', fontname='Segoe UI', fontsize=16, weight='bold')
    plt.xlabel('Дата', fontname='Segoe UI', fontsize=14)
    plt.ylabel('Сумма', fontname='Segoe UI', fontsize=14)
    plt.xticks(rotation=0)
    plt.legend()
    plt.tight_layout()

    st.pyplot(fig)

def get_cmap_colors(cmap_name, num_colors):
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, num_colors))
    print("Returned colors:", colors)
    return colors

def plot_financial_pie_charts(data):
    income_data = data[data['Доходы/Расходы'] == 'Доходы']
    expense_data = data[data['Доходы/Расходы'] == 'Расходы']

    def group_small_categories(series, threshold=0.05):
        small_categories = series[series / series.sum() < threshold]
        large_categories = series[series / series.sum() >= threshold]
        if not small_categories.empty:
            other_sum = small_categories.sum()
            large_categories = pd.concat([large_categories, pd.Series([other_sum], index=['Мелкие'])])
        return large_categories

    income_by_category = group_small_categories(income_data.groupby('Категория')['Сумма'].sum())
    expense_by_category = group_small_categories(expense_data.groupby('Категория')['Сумма'].sum())

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    font = FontProperties(family='Segoe UI', size=16)

    wedges1, texts1, autotexts1 = axs[0].pie(income_by_category, labels=income_by_category.index,
                                             autopct='%1.1f%%',
                                             startangle=140, radius=1.2,
                                             colors=get_cmap_colors(adjust_cmap('Blues', 0.2, 0.7), len(income_by_category)),
                                             textprops={'fontsize': 16, 'fontfamily': 'Segoe UI'})
    axs[0].set_title('Доходы по категориям', pad=70, fontdict={'fontweight': 'bold', 'family': 'Segoe UI', 'size': 16})

    wedges2, texts2, autotexts2 = axs[1].pie(expense_by_category, labels=expense_by_category.index,
                                             autopct='%1.1f%%',
                                             startangle=140, radius=1.2,
                                             colors=get_cmap_colors(adjust_cmap('Reds', 0.2, 0.7), len(income_by_category)),
                                             textprops={'fontsize': 16, 'fontfamily': 'Segoe UI'})
    axs[1].set_title('Расходы по категориям', pad=70, fontdict={'fontweight': 'bold', 'family': 'Segoe UI', 'size': 16})

    axs[0].legend(wedges1, income_by_category.index, prop=font, title="Категории", loc="center left", bbox_to_anchor=(1.1, 1))
    axs[1].legend(wedges2, expense_by_category.index, prop=font, title="Категории", loc="center right", bbox_to_anchor=(-0.1,1))

    plt.tight_layout()
    st.pyplot(fig)

def prepare_prompt(data):
    csv_string = data.to_string()
    return csv_string

def send_to_gpt_ai(prompt, section, country, social_status):
    api_key = 'sk-xxxx'
    if not api_key:
        return "API key not set. Please configure the environment variable."

    templates = [
            f'Твоя задача на основе полученных данных написать краткий обзор ПО ПУНКТАМ и покажи пользователю сколько он тратил на каждую категорию (в обзоре не нужны рекомендации, просто вкратце расскажи данные о тратах). Учитывай страну ({country}) и социальный статус пользователя ({social_status}).',
            f'Твоя задача на основе полученных данных дать по пунктам рекомендации экономии бюджета и точные расчеты моих данных где я мог сэкономить анализируя мои данные ({prompt}), мою страну и социальынй статус ({country, social_status}). Пользователю нужно короткая и понятная информация, как например (пример!): "вы слишком часто посещаете рестораны; Ходите 1 раз в неделю вместо 3 - Так вы сэкономите N тенге в месяц. На продукты у вас уходит N денег; В среднем в {country} можно уложиться в N в месяц." (ВСЁ ЭТО ЯВЛЯЕТСЯ ПРИМЕРОМ, составляй ответы так, чтобы они напрямую относились к банковской выписке). Также в ответе нужно больше цифр (жирным текстом) и не забывать про страну и социальный статус пользователя.',
        ]

    please = 'Используй больше цифр (данных из таблицы) - если ты показываешь проценты, то в скобках можешь провести расчет на основе банковской выписки. Составляй короткие рекомендации (не лонгрид). Не говори "возможно", просто прямо давай советы, которые необходимо соблюсти для экономии бюджета. Не забывай про страну и социальный статус пользователя'

    template = templates[section]
    content = template + " " +please

    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "assistant",
                 "content": "You are a financial helper"},
                {"role": "user", "content": content}
            ]
        )
        if response:
            return response.choices[0].message.content
        else:
            return "Ошибка"
    except Exception as e:
        return f"An error occurred: {e}"


def main():
    st.title("Финансовый аналитик")

    user_name = st.text_input("Введите ваше имя:", "")
    social_status = st.selectbox("Выберите ваш социальный статус:", ["Студент", "Работающий", "Пенсионер", "Безработный"])
    country = st.text_input("В какой стране вы находитесь?", "")

    uploaded_file = st.file_uploader("Загрузите свой CSV файл", type="csv")
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.write("Предварительный обзор ваших данных:")
        st.dataframe(data, width=700, height=200)

        if data is not None:
            prompt = prepare_prompt(data)

            if st.button("Проанализировать"):
                intro_analysis = send_to_gpt_ai(prompt, 0, country, social_status)
                st.subheader('Введение и обзор данных')
                plot_financial_data(data)
                st.write(intro_analysis)

                pie_chart_analysis = send_to_gpt_ai(prompt, 1, country, social_status)
                st.subheader('Рекомендации по категориям доходов и расходов')
                plot_financial_pie_charts(data)
                st.write(pie_chart_analysis)

if __name__ == "__main__":
    main()
