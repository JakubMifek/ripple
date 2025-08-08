python3 main.py ./samples/Baltazar.wav ./output_names/Baltazar.svg --smoothness 0.309 --color "black" --circle &
python3 main.py ./samples/Leontynka.wav ./output_names/Leontynka.svg --smoothness 0.309 --color "black" &
python3 main.py ./samples/Eliska.wav ./output_names/Eliska.svg --smoothness 0.309 --color "black" &
python3 main.py ./samples/Kuba.wav ./output_names/Kuba.svg --smoothness 0.309 --color "black" &
wait
python3 merge.py output_names