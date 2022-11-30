import os
import music21


def load_stream(datapath):
    """Loads midis from datapath and returns as stream"""
    all_midis= []
    for i in os.listdir(filepath):
        if i.endswith(".mid"):
            tr = os.path.join(filepath, i)
            midi = converter.parse(tr)
            all_midis.append(midi)
    return all_midis

def extract_notes(midi):
    """Extracts notes from midi file
    Returns note object
    """
    notes = []
    pick = None
    for j in midi:
        songs = instrument.partitionByInstrument(j)
        for part in songs.parts:
            pick = part.recurse()
            for element in pick:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append(".".join(str(n) for n in element.normalOrder))

    return notes

def clean_data(notes):
    """Eliminates rare notes.
    Notes that occur less then a 100 times
    """
    count_num = Counter(notes)

    #Getting a list of rare chords
    rare_notes = []
    for index, (key, value) in enumerate(count_num.items()):
        if value < 100:
            rare_notes.append(key)

    #Eleminating the rare notes
    for element in notes:
        if element in rare_notes:
            notes.remove(element)

    return notes
