# Movie Recommendation Chat

## Models Used

- Chat assistant: `Codex`, a GPT-5-based coding agent. Model used: GPT 5.4.
- Recommendation model used in the conversation: `SASRec_movielens` (`sequential`)
- Recommendation model was trained used the configuration found in `config/sasrec_mcp_ml1m.yml`.

## Transcript

### User

I've watched Toy Story, what's the next film I could watch? Recommend me at least 10 films and explain why you are suggesting each of them

### Assistant

Using the local `SASRec_movielens` sequential recommender with `Toy Story` as the seed, these are the strongest next-watch candidates. The explanations are my interpretation of why each is a good follow-up to `Toy Story`, not metadata returned by the model.

1. `Star Wars: Episode VI - Return of the Jedi`  
   A good next pick if you want another accessible adventure with humor, memorable characters, and a strong sense of payoff.

2. `Jurassic Park`  
   It keeps the big-audience blockbuster energy, but shifts from animation to suspense and spectacle without becoming too hard to follow.

3. `Braveheart`  
   This is a much more serious turn, but it fits if you want another emotionally direct, widely loved crowd-pleaser with strong momentum.

4. `Forrest Gump`  
   Recommended because it is similarly easy to connect with: warm, sentimental, funny at times, and built around a very likable central character.

5. `Star Wars: Episode V - The Empire Strikes Back`  
   A strong choice if what you want next is iconic adventure storytelling with richer stakes and excellent character chemistry.

6. `Back to the Future`  
   Probably one of the cleanest follow-ups here: playful, fast, funny, and built around the same kind of universally enjoyable storytelling that makes `Toy Story` work.

7. `Titanic`  
   Suggested as another hugely popular, emotionally immediate film that is easy to get pulled into even if it is very different in tone.

8. `The Godfather`  
   This is here as a prestige classic step-up: if you want to move from a familiar favorite into one of the most acclaimed mainstream films ever made.

9. `The Princess Bride`  
   A very natural recommendation if you liked the charm of `Toy Story`: witty, warm, adventurous, and full of personality.

10. `The Breakfast Club`  
    This one trades adventure for character interaction, but it works if you want something iconic, accessible, and driven by memorable voices.

If you want, I can also give you:

- a more `family-friendly` next-10,
- a more `adventure/sci-fi` next-10,
- or rerun the recommender with multiple films you've already seen.
