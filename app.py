import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.tools import Tool

# ======================================================
# CONFIGURAÇÃO DA PÁGINA
# ======================================================
st.set_page_config(
    page_title="Análise de Dados com IA",
    layout="wide"
)

st.title("📊 Análise de Dados com IA (LangChain + Streamlit)")

# ======================================================
# SIDEBAR — API KEY
# ======================================================
with st.sidebar:
    st.header("🔑 Configuração")
    openai_api_key = st.text_input(
        "Informe sua OpenAI API Key",
        type="password",
        help="A chave é usada apenas nesta sessão e não é armazenada."
    )

if not openai_api_key:
    st.warning("🔐 Insira sua OpenAI API Key para continuar.")
    st.stop()

os.environ["OPENAI_API_KEY"] = openai_api_key

# ======================================================
# UPLOAD DO ARQUIVO
# ======================================================
st.subheader("📂 Upload do arquivo Excel")

arquivo = st.file_uploader(
    "Envie um arquivo .xlsx ou .xls",
    type=["xlsx", "xls"]
)

if not arquivo:
    st.info("⬆️ Envie um arquivo Excel para iniciar a análise.")
    st.stop()

df = pd.read_excel(arquivo)

st.success("✅ Arquivo carregado com sucesso!")
st.dataframe(df.head())

# ======================================================
# LLM
# ======================================================
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# ======================================================
# FERRAMENTA PYTHON (EXECUÇÃO SOBRE O DF)
# ======================================================
python_tool = PythonAstREPLTool(
    locals={
        "df": df,
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns
    }
)

tool_python = Tool(
    name="Python",
    func=python_tool.run,
    description="""
    Use esta ferramenta para executar código Python sobre o dataframe `df`.
    Utilize pandas, numpy, matplotlib e seaborn.
    Gere gráficos quando solicitado.
    """
)

# ======================================================
# PROMPT DO AGENTE
# ======================================================
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Você é um analista de dados especialista.
Você tem acesso a um DataFrame pandas chamado `df`.

Regras:
- Use Python sempre que precisar calcular, filtrar ou criar gráficos.
- Para gráficos, use matplotlib ou seaborn.
- Não crie dados fictícios.
- Sempre responda em português.
- Seja claro e objetivo.
            """
        ),
        ("human", "{input}")
    ]
)

# ======================================================
# AGENTE
# ======================================================
agent = create_openai_tools_agent(
    llm=llm,
    tools=[tool_python],
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[tool_python],
    verbose=True
)

# ======================================================
# PERGUNTA DO USUÁRIO
# ======================================================
st.subheader("❓ Pergunta")

pergunta = st.text_area(
    "Faça uma pergunta sobre os dados:",
    placeholder="Ex: Qual é a média da coluna X? Gere um gráfico de Y por Z."
)

if st.button("🚀 Executar análise"):

    if not pergunta.strip():
        st.warning("Digite uma pergunta.")
        st.stop()

    with st.spinner("🤖 Analisando os dados..."):
        try:
            resposta = agent_executor.invoke(
                {"input": pergunta}
            )

            st.subheader("📌 Resposta")
            st.write(resposta["output"])

            # Exibir gráfico se existir
            fig = plt.gcf()
            if fig.get_axes():
                st.subheader("📈 Gráfico gerado")
                st.pyplot(fig)
                plt.clf()

        except Exception as e:
            st.error("❌ Ocorreu um erro durante a análise.")
            st.exception(e)
