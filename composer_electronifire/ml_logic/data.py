import os
import collections
import pandas as pd
import numpy as np
import pretty_midi as pm
import music21
from typing import List


def load_stream(datapath):
    """Loads midis from datapath and returns as stream"""
    all_midis= []
    for i in os.listdir('./raw_data'):
        if i.endswith(".mid"):
            tr = os.path.join(datapath, i)
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


### Help functions for multivariate model

def midis_set(datapath):
    """Loads midis from datapath and returns as list of PrettyMIDIS"""
    all_midis= []
    for i in os.listdir(datapath):
        if i.endswith(".mid"):
            tr = os.path.join(datapath, i)
            midi = pm.PrettyMIDI(tr)
            all_midis.append(midi)
    return all_midis

def midi_to_notes(midi) -> pd.DataFrame:
    """Turns midi into dataframe containing
    pitch, step, duration and velocity"""

    notes = collections.defaultdict(list)
    sorted_notes=[]
    for i in range(len(midi.instruments)):
        # Sort the notes by start time
        sorted_notes += list(midi.instruments[i].notes)

    sorted_notes = sorted(sorted_notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start

    for note in sorted_notes:
        start = note.start
        end = note.end
        notes['start'].append(start)
        notes['pitch'].append(int(note.pitch))
        notes['step'].append(start - prev_start)
        notes['duration'].append(end - start)
        notes['velocity'].append(note.velocity)
        prev_start = start

    df = pd.DataFrame({name: np.array(value) for name, value in notes.items()}).sort_values(by=['start'])
    return df

def notes_to_midi(notes: pd.DataFrame) -> pm.PrettyMIDI:
    """Turns dataframe containing pitch, step, duration and velocity
    into midi"""
    midi = pm.PrettyMIDI()
    instrument = pm.Instrument(program=pm.instrument_name_to_program('Acoustic Grand Piano'))

    prev_start = 0
    for i, note in notes.iterrows():
        start = float(prev_start + note['step'])
        end = float(start + 0.27)
        note = pm.Note(
            velocity=int(note['velocity']),
            pitch=int(note['pitch']),
            start=start,
            end=end,
        )
        instrument.notes.append(note)
        prev_start = start

    midi.instruments.append(instrument)
    return midi

def notes_to_chords(df):
    """group notes played at the same time into chords.
    returns dataframe with pitch, start and step columns"""

    df['pitch'] = df['pitch'].apply(lambda x: str(x))
    df = df[['pitch', 'start', 'step', 'velocity', 'duration']].\
        groupby('start', as_index=False).agg({'pitch':lambda x : '.'.join(x),
                                                'step':'sum',
                                                'velocity':'max',
                                                'duration':'max'
                                                })
    return df

def strongest_note(df):
    """group notes played at the sametime based on the stongest note played.
    returns dataframe with pitch, start and step columns"""

    df = df[['pitch', 'start', 'step', 'velocity', 'duration']].\
            groupby('start', as_index=False).\
            agg({'pitch':lambda g: g[df.loc[g.index]['velocity'].idxmax()],
                 'step':'sum',
                 'velocity':'max',
                 'duration':'max'})
    df['pitch'] = df['pitch']-21
    return df

def join_dfs(df_list):
    df = pd.concat(df_list).\
            reset_index().\
            drop(columns='index')
    return df
