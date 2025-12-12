import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain_core.output_parsers import StrOutputParser

# ======================================================
# STREAMLIT
# ======================================================
st.set_page_config(page_title="Análise de Dados com IA", layout="wide")
st.title("📊 Análise de Dados com IA")

arquivo = st.file_uploader("📂 Envie um Excel (.xlsx)", type=["xlsx"])
if not arquivo:
    st.stop()

df = pd.read_excel(arquivo)
st.success("Arquivo carregado!")

with st.expander("🔍 Visualizar dados"):
    st.dataframe(df.head(20))

# ======================================================
# LLM
# ======================================================
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# ======================================================
# TOOL PYTHON
# ======================================================
python_tool = PythonAstREPLTool(
    locals={
        "df": df,
        "pd": pd,
        "plt": plt,
        "sns": sns
    }
)

# ======================================================
# PROMPT
# ======================================================
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Você é um analista de dados especialista em pandas e visualização.

        Regras:
        - Use sempre o DataFrame `df`
        - Para cálculos ou tabelas, gere código Python
        - Para gráficos, use matplotlib ou seaborn
        - Sempre finalize gráficos com plt.show()
        - Responda em português
        """
    ),
    ("human", "{input}")
])

# ======================================================
# PIPELINE MODERNO (SEM AGENTEXECUTOR)
# ======================================================
chain = (
    prompt
    | llm.bind_tools([python_tool])
    | StrOutputParser()
)

# ======================================================
# UI
# ======================================================
st.subheader("💬 Faça sua pergunta")

pergunta = st.text_input(
    "Ex: Qual a média da coluna X? | Gere um gráfico da distribuição de Y"
)

if st.button("Executar") and pergunta:
    with st.spinner("Analisando..."):
        try:
            resposta = chain.invoke({"input": pergunta})

            st.subheader("📌 Resultado")
            st.write(resposta)

            for fig_num in plt.get_fignums():
                st.pyplot(plt.figure(fig_num))
            plt.close("all")

        except Exception as e:
            st.error("Erro na análise")
            st.exception(e)
