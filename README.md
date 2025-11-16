# StreamerFinder

Project Name: Stream Finder

Names and net ids of the students in the team: one per line

Javohir Abdurazzakov - ja688

Yung Hsueh Lee - yl2892

Joseph Abramov - ja653

Eli Furgeson - ehf38 

Idea: Our application will suggest streamers sorted by a match score based on key traits you enjoy. It will achieve this by web scraping the top/trending ~ 300 relevant tweets about him, 1 wikipedia page (if it exists), and top ~300 reddit posts that comes up when you search “[streamer name] reddit” on google for approximately 1000 streamers on platforms like Twitch, then identifying and ranking streamers mentioned alongside the traits you prefer.

Goals: The goal is to help users easily discover streamers that align with their interests by leveraging real-time social media trends, ultimately enhancing their viewing experience with personalized recommendations.

Information Retrieval Aspect: Before any user input, our system will pre-collect data by web scraping trending tweets, reddit posts, and wikipedia information related to a curated list of around 1000 streamers. This pre-scraped data is processed using natural language processing techniques to extract relevant keywords, key phrases, and sentiment indicators. When a user enters specific key phrases, the system will compare these phrases with the pre-processed data to compute similarity scores, allowing it to quickly and accurately rank streamers based on how closely they match the desired traits.

Social Information Aspect:

By aggregating and analyzing trending tweets and other online info like reddit and wikipedia, the app utilizes current social media sentiment and chatter as well as known factual information as a dynamic data source to assess and recommend streamers based on popularity and relevance.

Relation to prior projects and existing apps:

This project is entirely unique and it is not related to any of the pre-brainstorming survey ideas or previous years’ projects featured in the Hall of Fame. 




Instructions:

pip install -r requirements.txt

python [filename.py]
