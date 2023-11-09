import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development
import os

from analyze_conversation import *



st.set_page_config(
    page_title="Perspect Conversational Performance Dashboard",
    page_icon="✅",
    layout="wide",
)

# dashboard title
st.title("Perspect Conversational Performance Dashboard")

files = os.listdir("./meeting-transcripts")
chosen_file = st.selectbox("Select a meeting transcript file to analyze", list(files))
uploaded_file = open(f"./meeting-transcripts/{chosen_file}")
if uploaded_file is not None:
    
    conversation = ConversationAnalyzer(uploaded_file)

    kpi_dict = conversation.get_conversation_kpi()

    # create three columns
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    
    # fill in those three columns with respective metrics or KPIs
    kpi1.metric(
        label="Number of Speakers 🧑👧",
        value=int(kpi_dict["Number of Speakers"]),
    )

    kpi2.metric(
        label="Total Conversation Time ⏳",
        value=f"{round(kpi_dict['Total Conversation Time'], 2)} seconds",
    )

    kpi3.metric(
        label = "Average Pause Between Speakers",
        value=f"{round(kpi_dict['Average Pause Between Speakers'], 2)} seconds"
    )

    kpi4.metric(
        label = "Average Sentence Length",
        value=f"{round(kpi_dict['Average Sentence Length'], 2)} words"
    )

    st.markdown("### Break down of speaker time")
    speaker_times = conversation.get_speaker_dist()
    fig = px.pie(pd.DataFrame(data={"speaker": list(speaker_times.keys()), 
                                    "time": list(speaker_times.values())}),
                    names="speaker",
                    values="time")
    st.plotly_chart(fig)

    # top-level filters
    speaker_filter = st.selectbox("Select the Speaker", list(conversation.speakers))

    speaker_kpi = conversation.get_speaker_kpi(speaker_filter)

    kpi1, kpi2 = st.columns(2)

    kpi1.metric(
        label = "Words Per Minute",
        value = f"{round(speaker_kpi['words_per_minute'], 2)} words"
    )

    kpi2.metric(
        label = "User Sentiment Score",
        value = f"{round(speaker_kpi['sentiment_score'], 2)}"
    )

    st.markdown("### AI Powered Speaker Feedback")
    st.markdown(speaker_kpi["model_feedback"])
