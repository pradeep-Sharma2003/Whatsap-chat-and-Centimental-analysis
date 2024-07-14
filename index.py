import streamlit as st
import preprocessor, functions
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


st.sidebar.title('WhatsApp Chat Analyzer')

uploaded_file = st.sidebar.file_uploader("Choose a File")

if uploaded_file is None:
    st.markdown('''Please export your WhatsApp chat (without media), whether it be a group chat or an individual/private chat, then click on "Browse Files" and upload it to this platform.''')
    st.markdown('''Afterward, kindly proceed to click on the "Analyse" button. This action will generate a variety of insights concerning your conversation.''')
    st.markdown(''' You will have the option to select the type of analysis, whether it is an overall analysis or one that specifically focuses on particular participants' analysis.''')
    st.markdown('Thank You!')
    st.markdown('Pradeep Sharma')

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()

    # st.write(bytes_data)

    # converting bytes into text
    data = bytes_data.decode('utf-8')

    # show text data
    # st.text(data)

    # DataFrame
    df = preprocessor.preprocess(data)

    # show dataframe
    # st.dataframe(df)

    # fetch unique user
    user_details = df['user'].unique().tolist()
    # remove Group Notifications
    if 'Group Notification' in user_details:
        user_details.remove('Group Notification')
    # sorting list
    user_details.sort()
    # insert overall option
    user_details.insert(0, 'OverAll')

    # drop down to select user
    selected_user = st.sidebar.selectbox('Show Analysis as:', user_details)

    if st.sidebar.button('Analyse'):

        num_msgs, num_med, link = functions.fetch_stats(selected_user, df)

        # overall statistics
        st.title('OverAll Basic Statistics')
        col1, col2, col3, = st.columns(3)
        with col1:
            st.header('Messages')
            st.subheader(num_msgs)
        with col2:
            st.header('Media Shared')
            st.subheader(num_med)
        with col3:
            st.header('Link Shared')
            st.subheader(link)

        # monthly timeline
        timeline = functions.monthly_timeline(selected_user, df)
        st.title('Monthly Timeline')

        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['msg'], color='maroon')
        plt.xticks(rotation=90)
        st.pyplot(fig)

        # daily timeline
        timeline = functions.daily_timeline(selected_user, df)
        st.title('Daily Timeline')
        fig, ax = plt.subplots()
        ax.plot(timeline['date'], timeline['msg'], color='purple')
        plt.xticks(rotation=90)
        st.pyplot(fig)


        # active map
        st.title('Activity Map')
        col1, col2 = st.columns(2)

        active_month_df, month_list, month_msg_list, active_day_df, day_list, day_msg_list = functions.activity_map(selected_user, df)
        with col1:
            # active month
            st.header('Most Active Month')
            fig, ax = plt.subplots()
            ax.bar(active_month_df['month'], active_month_df['msg'])
            ax.bar(month_list[month_msg_list.index(max(month_msg_list))], max(month_msg_list), color='green', label = 'Highest')
            ax.bar(month_list[month_msg_list.index(min(month_msg_list))], min(month_msg_list), color='red', label = 'Lowest')
            plt.xticks(rotation=90)
            st.pyplot(fig)

        with col2:
            # active day
            st.header('Most Active Day')
            fig, ax = plt.subplots()
            ax.bar(active_day_df['day'], active_day_df['msg'])
            ax.bar(day_list[day_msg_list.index(max(day_msg_list))], max(day_msg_list), color='green', label='Highest')
            ax.bar(day_list[day_msg_list.index(min(day_msg_list))], min(day_msg_list), color='red', label='Lowest')
            plt.xticks(rotation=90)
            st.pyplot(fig)


        # most chatiest user
        if selected_user == 'OverAll':
            st.title('Most Active Users')

            x, percent = functions.most_chaty(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)
            with col1:
                ax.bar(x.index, x, color='cyan')

                st.pyplot(fig)
            with col2:
                st.dataframe(percent)

        # WordCloud
        df_wc = functions.create_wordcloud(selected_user, df)
        st.title('Most Common Words')

        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        
        # sentimental analysis
        st.markdown('<h2 class="section-title">Sentiment Analysis</h2>', unsafe_allow_html=True)
        sentiments = SentimentIntensityAnalyzer()
        positive = [sentiments.polarity_scores(i)["pos"] for i in df['msg']]
        negative = [sentiments.polarity_scores(i)["neg"] for i in df['msg']]
        neutral = [sentiments.polarity_scores(i)["neu"] for i in df['msg']]
        new_df = pd.DataFrame({'positive': positive, 'negative': negative, 'neutral': neutral})

        average_sentiments = new_df.mean()
        fig, ax = plt.subplots(figsize=(8, 6))
        average_sentiments.plot(kind='bar', color=['green', 'red', 'blue'], ax=ax)
        plt.xlabel('Sentiment')
        plt.ylabel('Average Value')
        plt.title('Average Sentiment Distribution')
        plt.xticks(rotation=0)
        st.pyplot(fig)

# Main content container
st.markdown('<div class="main">', unsafe_allow_html=True)
# Add content and functionalities here...
st.markdown('</div>', unsafe_allow_html=True)
