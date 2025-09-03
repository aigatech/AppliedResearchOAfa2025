from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from random import choice, choices

chord_list = ("A", "B", "C", "D", "E", "F", "G")
modifier_list = ("", "min", "maj7", "min7", "b", "sus4", "add9")
valid_chord_list = ('A', 'Amin', 'Amaj7', 'Amin7', 'Ab', 'Asus4', 'Aadd9',
                    'B', 'Bmin', 'Bmaj7', 'Bmin7', 'Bb', 'Bsus4', 'Badd9',
                    'C', 'Cmin', 'Cmaj7', 'Cmin7', 'Cb', 'Csus4', 'Cadd9',
                    'D', 'Dmin', 'Dmaj7', 'Dmin7', 'Db', 'Dsus4', 'Dadd9',
                    'E', 'Emin', 'Emaj7', 'Emin7', 'Eb', 'Esus4', 'Eadd9',
                    'F', 'Fmin', 'Fmaj7', 'Fmin7', 'Fb', 'Fsus4', 'Fadd9',
                    'G', 'Gmin', 'Gmaj7', 'Gmin7', 'Gb', 'Gsus4', 'Gadd9')
song_sections = ("intro", "verse", "chorus", "solo", "bridge")

def seq_gen(instruction):
    model = AutoModelForCausalLM.from_pretrained("xiejoshua/chordgpt", device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained("sgugger/gpt2-like-tokenizer", padding_side="left")

    # Generates a random first chord for variability, with weights for feasibility.
    model_inputs = tokenizer([f"<{instruction}_1> {choice(chord_list)}{choices(modifier_list, [0.4, 0.1, 0.1, 0.1, 0.2, 0.05, 0.05])[0]}"], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, pad_token_id=tokenizer.eos_token_id)
    output = tuple(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].split())
    chords_added = 0
    chords_to_add = []
    for chord in output:
        if chords_added <= 4:
            if chord in valid_chord_list:
                chords_to_add.append(chord)
            chords_added += 1
        else:
            break
    return chords_to_add

def seq_gen_main():
    continue_running = True
    while continue_running:
        print("""-----------------------------------------------------
RANDOM SEQUENCE GENERATOR
-----------------------------------------------------
You have chosen to generate a random four-chord
sequence. Please select whether you would like to 
generate an INTRO, VERSE, CHORUS, SOLO, or BRIDGE.
-----------------------------------------------------""")
        instruction = input("(###) Enter your selection: ").lower()
        if instruction in song_sections:
            gen = seq_gen(instruction)
            plaintext_chords = ""
            for chord in gen:
                plaintext_chords += f"{chord} "
            print(f"-----------------------------------------------------\n(!!!) YOUR {instruction.upper()} SEQUENCE: {plaintext_chords}\n-----------------------------------------------------")
            continue_running = False
            time.sleep(1)
        else:
            print("-----------------------------------------------------\n(???) That doesn't seem to be a valid input. Try again!\n-----------------------------------------------------")
            time.sleep(1)

def song_gen(intro):
    plaintext_intro_chords = ""
    for chord in intro:
        plaintext_intro_chords += f"{chord} "
    input = f"<intro_1> {plaintext_intro_chords} "
    outputchords = {"verse": [], "chorus":[], "solo": [], "bridge": []}
    model = AutoModelForCausalLM.from_pretrained("xiejoshua/chordgpt", device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained("sgugger/gpt2-like-tokenizer", padding_side="left")

    for i in range(1, 5):
        input += f"<{song_sections[i]}_1> "
        model_inputs = tokenizer(
            [input],
            return_tensors="pt").to(model.device)

        generated_ids = model.generate(**model_inputs, pad_token_id=tokenizer.eos_token_id)
        output = tuple(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].split(f"<{song_sections[i]}_1>", 1)[1])
        chords_added = 0
        chords_to_add = []
        for chord in output:
            if chords_added <= 4:
                if chord in valid_chord_list:
                    chords_to_add.append(chord)
                    input += f"{chord} "
                    chords_added += 1
            else:
                break
        outputchords[song_sections[i]] = chords_to_add
    outputchords["intro"] = intro
    return outputchords

def song_gen_main():
    continue_running = True
    while continue_running:
        print("""-----------------------------------------------------
RANDOM SONG GENERATOR
-----------------------------------------------------
You have chosen to generate a random song made of
four-chord sequences. We will start by generating
four sequences for your INTRO. You will then 
select your favorite, which the rest of your song 
will be based on.
-----------------------------------------------------""")
        seq_list = []
        for i in range(4):
            seq_list.append(seq_gen("intro"))
            plaintext_chords = ""
            for chord in seq_list[i]:
                plaintext_chords += f"{chord} "
            print(f"{i+1}. {plaintext_chords}")
        print("-----------------------------------------------------")
        instruction = input("(###) Enter your selection: ").lower()
        try:
            if 0 < int(instruction) < 5:
                output = song_gen(seq_list[int(instruction)-1])
                print("-----------------------------------------------------")
                for section in output.keys():
                    plaintext_chords = ""
                    for chord in output[section]:
                        plaintext_chords += f"{chord} "
                    print(f"{section.upper()}: {plaintext_chords}")
                print("-----------------------------------------------------")
                continue_running = False
                time.sleep(1)
            else:
                print("-----------------------------------------------------\n(???) That doesn't seem to be a valid input. Try again!\n-----------------------------------------------------")
                time.sleep(1)
        except:
            print("-----------------------------------------------------\n(???) That doesn't seem to be a valid input. Try again!\n-----------------------------------------------------")
            time.sleep(1)

def main():
    continue_running = True
    while continue_running:
        print("""WELCOME TO FOUR-CHORD GENERATOR, POWERED BY CHORDGPT!
-----------------------------------------------------
1: Random Sequence Generator
2: Random Song Generator
3: Exit
-----------------------------------------------------""")
        instruction = input("(###) Enter your selection: ")
        #try:
        instruction = int(instruction)
        if instruction == 1:
            seq_gen_main()
        elif instruction == 2:
            song_gen_main()
        elif instruction == 3:
            print("-----------------------------------------------------\n(***) We hope you enjoyed Four-Chord Generator. Until next time!")
            continue_running = False
        else:
            print("-----------------------------------------------------\n(???) That doesn't seem to be a valid input. Try again!\n-----------------------------------------------------")
            time.sleep(1)
        #except:
        #    print("-----------------------------------------------------\n(???) That doesn't seem to be a valid input. Try again!\n-----------------------------------------------------")
        #    time.sleep(1)

main()