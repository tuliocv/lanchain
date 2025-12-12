import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ======================================================
# CONFIGURAÇÃO DA PÁGINA
# ======================================================
st.set_page_config(
    page_title="Análise de Dados com IA",
    layout="wide"
)

st.title("📊 Análise de Dados com IA (Streamlit + LangChain)")

# ======================================================
# SIDEBAR — API KEY
# ======================================================
with st.sidebar:
    st.header("🔑 OpenAI")
    api_key = st.text_input(
        "Informe sua OpenAI API Key",
        type="password",
        help="A chave é usada apenas durante a sessão."
    )

if not api_key:
    st.warning("Insira sua OpenAI API Key para continuar.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

# ======================================================
# UPLOAD DO ARQUIVO
# ======================================================
st.subheader("📂 Upload do Excel")

arquivo = st.file_uploader(
    "Envie um arquivo Excel (.xlsx ou .xls)",
    type=["xlsx", "xls"]
)

if not arquivo:
    st.info("Envie um arquivo para iniciar.")
    st.stop()

df = pd.read_excel(arquivo)

st.success("Arquivo carregado com sucesso!")
st.dataframe(df.head())

# ======================================================
# LLM
# ======================================================
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# ======================================================
# PROMPT — GERADOR DE CÓDIGO PYTHON
# ======================================================
prompt = ChatPromptTemplate.from_template("""
Você é um analista de dados especialista.

Você tem acesso a um DataFrame pandas chamado `df`.

Colunas disponíveis:
{colunas}

Amostra dos dados:
{amostra}

Pergunta do usuário:
{pergunta}

Regras obrigatórias:
- Gere APENAS código Python válido
- Use pandas, numpy, matplotlib ou seaborn
- NÃO faça importações
- NÃO use markdown
- Se gerar gráfico, use matplotlib ou seaborn
- O DataFrame já existe como `df`

Código Python:
""")

cadeia = prompt | llm | StrOutputParser()

# ======================================================
# PERGUNTA DO USUÁRIO
# ======================================================
st.subheader("❓ Pergunta")

pergunta = st.text_area(
    "Pergunte algo sobre os dados:",
    placeholder="Ex: Gere um gráfico da média de vendas por categoria"
)

if st.button("🚀 Executar análise"):

    if not pergunta.strip():
        st.warning("Digite uma pergunta.")
        st.stop()

    with st.spinner("🤖 Analisando..."):

        colunas = "\n".join([f"- {c} ({t})" for c, t in df.dtypes.items()])
        amostra = df.head(5).to_dict(orient="records")

        codigo = cadeia.invoke({
            "colunas": colunas,
            "amostra": amostra,
            "pergunta": pergunta
        })

        # Limpeza de segurança
        codigo = codigo.replace("```python", "").replace("```", "").strip()

        st.subheader("🧠 Código gerado pela IA")
        st.code(codigo, language="python")

        # Execução controlada
        exec_context = {
            "df": df,
            "pd": pd,
            "np": np,
            "plt": plt,
            "sns": sns
        }

        try:
            exec(codigo, exec_context)

            # Exibir gráfico se existir
            fig = plt.gcf()
            if fig.get_axes():
                st.subheader("📈 Gráfico")
                st.pyplot(fig)
                plt.clf()

            st.success("Análise concluída com sucesso!")

        except Exception as e:
            st.error("Erro ao executar o código gerado.")
            st.exception(e)
