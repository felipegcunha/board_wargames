import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import cv2
import pytesseract
from PIL import Image
import numpy as np

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# Verifique se o Tesseract está acessível
print(pytesseract.get_tesseract_version())

def extract_text_from_image(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    custom_config = r'--oem 3'
    text = pytesseract.image_to_string(gray)
    text_list = text.split()
    text_list = [word.strip().lower() for word in text_list]
    
    try:
        idx_start = len(text_list) - 1 - next((i for i, word in enumerate(reversed(text_list)) if word.startswith("Meta")), -1)
        text_list = text_list[idx_start + 1:]
    except ValueError:
        pass
    
    text_list = [word for word in text_list if word.isdigit() or len(word) >= 3]
    
    color_keywords = ["Red", "Blue", "Green", "Yellow", "Purple", "Orange"]
    print("Texto extraído:", text_list)
    num_jogadores = next((i for i, word in enumerate(text_list) if word in color_keywords), len(text_list))
    
    return text_list, num_jogadores

def clean_numeric_values(value):
    if isinstance(value, str):
        return value.replace(',', '').strip()
    return value

def parse_text_to_dataframe(text_list, num_jogadores):
    data = []
    total_required = num_jogadores * 7
    
    if len(text_list) < total_required:
        st.warning(text_list)
        st.warning("O número de elementos extraídos da imagem é menor do que o esperado. Verifique a imagem e tente novamente.")
        return pd.DataFrame(columns=["Nome", "Time", "Kills", "Assistências", "Dano", "Dano Sofrido", "Cura"])
    
    for i in range(num_jogadores):
        try:
            row = [
                text_list[i],
                text_list[num_jogadores + i],
                text_list[2 * num_jogadores + i],
                text_list[3 * num_jogadores + i],
                clean_numeric_values(text_list[4 * num_jogadores + i]),
                clean_numeric_values(text_list[5 * num_jogadores + i]),
                clean_numeric_values(text_list[6 * num_jogadores + i])
            ]
            data.append(row)
        except IndexError:
            st.error(f"Erro ao processar os dados do jogador {i + 1}. Verifique a integridade dos dados extraídos.")
            break
    
    df = pd.DataFrame(data, columns=["Nome", "Time", "Kills", "Assistências", "Dano", "Dano Sofrido", "Cura"])
    
    for col in ["Kills", "Assistências", "Dano", "Dano Sofrido", "Cura"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def calculate_mvp(df, weights):
    df["MVP Score"] = (
        df["Kills"] * weights["Kills"] +
        df["Assistências"] * weights["Assistências"] +
        df["Dano"] * weights["Dano"] -
        df["Dano Sofrido"] * weights["Dano Sofrido"] +
        df["Cura"] * weights["Cura"]
    )
    mvp = df.loc[df["MVP Score"].idxmax()]
    return mvp

def plot_radar_chart(df, selected_players):
    df = df[df["Nome"].isin(selected_players)]
    categorias = ["Dano", "Dano Sofrido", "Cura"]
    pesos = {"Dano": 0.1, "Dano Sofrido": 0.1, "Cura": 0.2}
    
    fig = go.Figure()
    for _, row in df.iterrows():
        valores = [row[col] * pesos[col] for col in categorias]
        valores.append(valores[0])  
        fig.add_trace(go.Scatterpolar(
            r=valores,
            theta=categorias + [categorias[0]],
            fill='toself',
            name=row["Nome"]
        ))
    
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True, title="Dano, Dano Sofrido e Cura")
    st.plotly_chart(fig, use_container_width=True)

def plot_bar_chart(df, selected_players):
    df = df[df["Nome"].isin(selected_players)]
    fig = px.bar(df, x="Nome", y=["Kills", "Assistências"], barmode='group', title="Kills e Assistências")
    st.plotly_chart(fig, use_container_width=True)

def main():
    df = pd.DataFrame()
    st.sidebar.header("Filtros")
    uploaded_files = st.file_uploader("Carregar Arquivos (CSV ou Imagem)", type=["csv", "png", "jpg", "jpeg"], accept_multiple_files=True)
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.type == "text/csv":
                temp_df = pd.read_csv(uploaded_file)
            else:
                image = Image.open(uploaded_file)
                text_list, num_jogadores = extract_text_from_image(image)
                temp_df = parse_text_to_dataframe(text_list, num_jogadores)
            
            df = pd.concat([df, temp_df], ignore_index=True)
    
    st.sidebar.subheader("Opções de Exibição")
    view_option = st.sidebar.radio("Escolha a exibição:", ["Tabela", "Comparação", "MVP da Partida"])
    
    if not df.empty:
        if view_option == "Tabela":
            st.subheader("📋 Tabela de Estatísticas")
            st.dataframe(df)
        elif view_option == "MVP da Partida":
            st.subheader("🏆 MVP da Partida")
            weights = {"Kills": 5, "Assistências": 3, "Dano": 0.1, "Dano Sofrido": -0.05, "Cura": 0.2}
            mvp = calculate_mvp(df, weights)
            st.write(f"**MVP: {mvp['Nome']}**")
            st.write(f"Com {mvp['Kills']} kills, {mvp['Assistências']} assistências e {mvp['Dano']} de dano causado, {mvp['Nome']} foi o destaque da partida!")
            st.write("A pontuação do MVP foi calculada com a seguinte fórmula: Kills x 5 + Assistências x 3 + Dano x 0.1 - Dano Sofrido x 0.05 + Cura x 0.2")
        else:
            st.subheader("📊 Comparação de Estatísticas")
            selected_players = st.sidebar.multiselect("Selecione jogadores para exibição", df["Nome"].unique(), default=df["Nome"].unique())
            plot_radar_chart(df, selected_players)
            plot_bar_chart(df, selected_players)

if __name__ == "__main__":
    main()
