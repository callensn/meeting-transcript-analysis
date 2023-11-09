import json
import numpy as np
import openai
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 

# uncomment this and include your openai account api key 
# openai.api_key = 

class ConversationAnalyzer:
    def __init__(self, json_file):
        # sentiment analysis
        self.sentiment_analysis = SentimentIntensityAnalyzer()

        # load convo to json
        self.convo = json.load(json_file)
        
        # get speakers
        self.speakers = set()
        for sent in self.convo:
            self.speakers.add(sent["speaker"])

    def stitch_sentence(self, s):
        words = s["words"]
        return  " ".join([f"{s['speaker']}: "] +\
                        [word["text"] for word in words])
                        # [f"| TIME ELAPSED: {round(total_time_elapsed, 2)} sec"] + \
                        # [f"| WORDS PER SECOND: {round(len(words) / total_time_elapsed, 2)}"] + \
                        # [f"| WORDS PER SECOND: {round(len(words) / total_time_elapsed, 2)}"])

    def stitch_convo(self):
        thread = []
        for sent in self.convo:
            thread.append(self.stitch_sentence(sent))
        return "\n".join(thread)

    def total_time_elapsed(self):
        stop_time = float(self.convo[-1]["words"][-1]["end_timestamp"])
        start_time = float(self.convo[0]["words"][0]["start_timestamp"] )
        return stop_time - start_time

    def get_conversation_kpi(self):
        return {"Total Conversation Time": self.total_time_elapsed(),
                "Number of Speakers": len(self.speakers),
                "Average Sentence Length": np.average([len(s["words"]) for s in self.convo]),
                "Average Pause Between Speakers": self.get_avg_pause()}
        
    def get_avg_pause(self):
        pauses = []
        for i in range(1, len(self.convo)):
            start = float(self.convo[i]["words"][0]["start_timestamp"])
            stop = float(self.convo[i-1]["words"][-1]["end_timestamp"])
            pauses.append(start - stop)
        return sum(pauses) / len(pauses)
        
    def get_speaker_dist(self):
        speaker_times = {s: 0 for s in self.speakers}
        for sent in self.convo:
            words = sent["words"]
            tot_time = float(words[-1]["end_timestamp"]) - float(words[0]["start_timestamp"])
            speaker_times[sent["speaker"]] += tot_time
        return speaker_times

    def get_speaker_kpi(self, speaker):
        # 1. sentiment analysis
        sentences = []
        total_time_talking = 0
        total_words = 0
        for sent in self.convo:
            if sent["speaker"] == speaker:
                sentences.append(self.stitch_sentence(sent))
                total_time_talking += float(sent["words"][-1]["end_timestamp"]) - float(sent["words"][0]["start_timestamp"])
                total_words += len(sent["words"])

        # 2. speaker words per minute
        words_per_minute = (total_words / total_time_talking) * 60

        sentiment_scores = []
        for sent in sentences:
            sentiment_scores.append(self.sentiment_analysis.polarity_scores(sent)['compound'])
        sentiment_score = np.average(sentiment_scores)
        

        # 3. llm feedback
        full_conversation = self.stitch_convo()
        prompt = f"Based on the preceding conversation, does {speaker} have effective communication as a soft skill? Additionally, what types of soft skills does {speaker} possess?"
        model_input = [{"role": "user", "content": full_conversation}]
        feedback = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=model_input)

        return {"sentiment_score": sentiment_score,
                "words_per_minute": words_per_minute,
                "model_feedback": feedback.choices[0].message.content}






    

            

