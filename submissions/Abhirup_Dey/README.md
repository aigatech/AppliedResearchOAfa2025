\# VibeFeeler — Deterministic Emotion Bars + Style Replies (Abhirup Dey)



VibeFeeler is a tiny and somewhat predictable Hugging Face project:

1\) Uses a HF model to detect top emotions and draws ASCII bars that display the top emotions.

2\) Produces a deterministic TL;DR depending on how long the input message is (extractive, no generation)

3\) Returns a deterministic style reply (`coach`, `genz`, `pirate`, `formal`)

&nbsp;	a. Use the coach for any assistance related messages (I need help on my English Test, etc.)

&nbsp;	b. Use the pirate one for anything

&nbsp;	c. Use the genz one to talk to as a friend (fighting, playing, etc)

&nbsp;	d. Use the formal one for anything, preferably as your partner for some sort of project





\## Hugging Face model

\- Emotions (multi-label): `joeddav/distilbert-base-uncased-go-emotions-student`



\## How to run (Windows)

```bat

python -m venv .venv

.venv\\Scripts\\activate

pip install -r submissions\\Abhirup\_Dey\\requirements.txt



SAMPLE INPUTS (note, if you want to make your own personalized messages, please right them out in proper English and try keeping the input short; I do not guarantee that the peronsalized response for the models will work properly every time (for example the coach might not output the message in grammatically correct English but it will output something at the very least loosely related), but the emotion detection/classification will properly work virtually always, and there will always be an output message. Btw, the pirate works the best, that's my proudest one :D):



\# Coach (default) \[you do not need the --style coach in the code line while running it in terminakl, but it is recommended)

python submissions\\Abhirup\_Dey\\main.py "I failed my English test badly; can you help me next time?" --style coach



output:

=== INPUT ===

I failed my English Test really badly; can you help me tomorrow?



=== EMOTIONS ===

Disappointment 0.12  ██------------------

Curiosity    0.11  ██------------------

Remorse      0.11  ██------------------

Confusion    0.09  ██------------------

Caring       0.07  █-------------------



=== TL;DR ===

I failed my English Test really badly; can you help me tomorrow?



=== REPLY (coach) ===

Yes—we'll work on your English test tomorrow. First, we'll review mistakes and drill the weak spots.







\# Gen-Z (deterministic)

python submissions\\Abhirup\_Dey\\main.py "I am going to fight you because you annoy me, please stop." --style genz



output:



=== INPUT ===

I am going to fight you because you annoy me, please stop.



=== EMOTIONS ===

Annoyance    0.25  █████---------------

Anger        0.21  ████----------------

Disapproval  0.10  ██------------------

Disappointment 0.07  █-------------------

Confusion    0.05  █-------------------



=== TL;DR ===

I am going to fight you because you annoy me, please stop.



=== REPLY (genz) ===

i am gonna fight you because i annoy you fr ✨



\# Pirate

python submissions\\Abhirup\_Dey\\main.py "Thanks for always having my back, I will always think of you as my one true friend" --style pirate




Output:

=== INPUT ===

Thanks for always having my back, I will always think of you as my one true friend.



=== EMOTIONS ===

Gratitude    0.27  █████---------------

Admiration   0.12  ██------------------

Caring       0.11  ██------------------

Approval     0.09  ██------------------

Pride        0.08  ██------------------



=== TL;DR ===

Thanks for always having my back, I will always think of you as my one true friend.



=== REPLY (pirate) ===

Arrr, ye be true crew, matey—stand fast together.




\# Formal



python submissions\\Abhirup\_Dey\\main.py "We have to finish this as soon as possible. Do you want us to fail???" --style formal





output:



=== INPUT ===

We have to finish this as soon as possible. Do you want us to fail???



=== EMOTIONS ===

Confusion    0.14  ███-----------------

Desire       0.10  ██------------------

Disapproval  0.08  ██------------------

Curiosity    0.08  ██------------------

Nervousness  0.05  █-------------------



=== TL;DR ===

We have to finish this as soon as possible. Do you want us to fail???



=== REPLY (formal) ===

We need to finish this.



