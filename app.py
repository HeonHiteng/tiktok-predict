import gradio as gr
import requests
import os
import pickle

headers = {
	"X-RapidAPI-Key": os.getenv('TIKTOK_API'),
	"X-RapidAPI-Host": "tiktok-scraper7.p.rapidapi.com"
}

with open('model_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('model_random_forest.pkl', 'rb') as f:
    model = pickle.load(f)

def get_video_details(link):
    response = requests.get("https://tiktok-scraper7.p.rapidapi.com/", headers=headers, params={"url":link,"hd":"1"})
    video_info = response.json()
    length = video_info['data']['duration']
    video_like = int(video_info['data']['digg_count'])
    video_share = int(video_info['data']['share_count'])
    video_comment = int(video_info['data']['share_count'])
    video_view = int(video_info['data']['comment_count'])
    video_engagement = video_like+video_share+video_comment+video_view
    play_url = video_info['data']['play']
    user_id = video_info['data']['author']['id']
    music_id = video_info['data']['music_info']['id']
    create_time = video_info['data']['create_time']

    response = requests.get("https://tiktok-scraper7.p.rapidapi.com/user/info", headers=headers, params={"user_id":user_id})
    user_info = response.json()

    follower = user_info['data']['stats']['followerCount']
    total_likes = user_info['data']['stats']['heartCount']
    total_video = user_info['data']['stats']['videoCount']

    markdown_for_showing = f"""**Video Information**
- **Likes**: {video_like}
- **Shares**: {video_share}
- **Comments**: {video_comment}
- **Views**: {video_view}
"""
    html_to_play = f'<video src={play_url} />'
    return html_to_play,markdown_for_showing, length, follower, total_likes, total_video, gr.update(visible=True)

def visible_component(component):
    return gr.update(visible=True)

def predict(length,followers,total_likes,total_videos,days_since_debut):
    data = scaler.transform([[length,followers,total_likes,total_videos,days_since_debut]])
    return model.predict(data)

css = """
button {
    height: 100%;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Row(variant='compact',equal_height=True):
        with gr.Column(scale=4):
            video_url = gr.Textbox(label="Enter a Tiktok Video URL",placeholder="https://www.tiktok.com/...")
        with gr.Column(scale=1):
            submit_button = gr.Button("Get Data from URL")
    second_row = gr.Row(variant='compact',equal_height=True, visible=False)
    with second_row:
        with gr.Column(scale=1):
            tiktok_video = gr.HTML()
            video_info = gr.Markdown()
        with gr.Column(scale=2):
            length = gr.Number(label="Length of video")
            follower = gr.Number(label="Follower Count")
            total_likes = gr.Number(label="User Total Likes")
            total_video = gr.Number(label="User Total Videos")
            days_since = gr.Number(label="Days since the music first debut")
    third_row = gr.Row(variant='compact',equal_height=True, visible=False)
    with third_row:
        predict_button = gr.Button("Predict")
        prediction = gr.Number(label="Predicted Likes for the video",visible=False)

    submit_button.click(fn=visible_component,
                        inputs=length,
                        outputs=second_row).then(fn=get_video_details, 
                        inputs=video_url, 
                        outputs=[
                            tiktok_video,
                            video_info,
                            length,
                            follower,
                            total_likes,
                            total_video,
                            third_row
                            ])
    predict_button.click(fn=predict,
                         inputs=[
                             length,
                             follower,
                             total_likes,
                             total_video,
                             days_since
                        ],
                        outputs=prediction).then(fn=visible_component,
                                                 inputs=length,
                                                 outputs=prediction)
demo.launch()   