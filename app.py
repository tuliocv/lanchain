import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.tools.python.tool import PythonAstREPLTool

# ======================================================
# CONFIGURAÇÃO STREAMLIT
# ======================================================
st.set_page_config(
    page_title="Análise de Dados com IA",
    layout="wide"
)

st.title("📊 Análise de Dados com IA")
st.write(
    "Envie uma planilha Excel e faça perguntas em linguagem natural. "
    "O assistente irá gerar análises, tabelas e gráficos automaticamente."
)

# ======================================================
# UPLOAD DO ARQUIVO
# ======================================================
arquivo = st.file_uploader(
    "📂 Envie um arquivo Excel (.xlsx)",
    type=["xlsx"]
)

if not arquivo:
    st.info("Envie um arquivo Excel para iniciar a análise.")
    st.stop()

df = pd.read_excel(arquivo)

st.success("Arquivo carregado com sucesso!")

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

tools = [python_tool]

# ======================================================
# PROMPT DO AGENTE
# ======================================================
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Você é um analista de dados especialista em pandas e visualização.

        Regras obrigatórias:
        - Sempre use o DataFrame chamado `df`
        - Para cálculos e tabelas, gere código Python
        - Para gráficos, use matplotlib ou seaborn
        - Finalize gráficos com plt.show()
        - Não invente nomes de colunas
        - Responda em português
        """
    ),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# ======================================================
# AGENTE
# ======================================================
agent = create_openai_tools_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False
)

# ======================================================
# INTERAÇÃO
# ======================================================
st.subheader("💬 Faça sua pergunta")

pergunta = st.text_input(
    "Exemplos: "
    "Qual a média da coluna X? | "
    "Crie uma tabela com a soma de vendas por categoria | "
    "Gere um gráfico da distribuição de idade"
)

if st.button("Executar análise") and pergunta:
    with st.spinner("Analisando os dados..."):
        try:
            resposta = executor.invoke({"input": pergunta})

            st.subheader("📌 Resultado")
            st.write(resposta["output"])

            # Renderizar gráficos
            for fig_num in plt.get_fignums():
                st.pyplot(plt.figure(fig_num))

            plt.close("all")

        except Exception as e:
            st.error("Erro ao executar a análise.")
            st.exception(e)
