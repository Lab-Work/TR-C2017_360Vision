ffmpeg -r 30 -i az_0726_expB/original_video.mp4 -vf crop=3840:80:0:468 -ss 00:01:22 -t 00:07:22 -r 30 az_0726_expB/edited_video.mp4 # Trim and cut raw video footgae
ffmpeg -i az_0726_expB/edited_video.mp4 -vframes 1 az_0726_expB/init_frame.png # Extract first frame
mkdir az_0726_expB/backgrounds # Make directory to store background images
python gen_bg.py az_0726_expB/ # Generate background images
cp az_0726_expB/backgrounds/background_c00.png az_0726_expB/background.png # Extract the first background image
python sub_bg.py az_0726_expB/ # Subtract backgrounds from every frames
python track.py az_0726_expB/ 12 # Track vehicles by template matching
mkdir data/ # Make directory to store smoothed data
python proc_tk.py az_0726_expB/ # Smooth vehicle trajectories
mkdir visual/ # Make directory to store data visualization
python vis_tk.py az_0726_expB/ # Generate data visualization
ffmpeg -i visual/demo_0726_expB.avi -pix_fmt yuv420p visual/demo_0726_expB.mp4 # Compress video size
rm visual/*.avi # Clear duplicates