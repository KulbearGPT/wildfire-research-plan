#!/usr/bin/env python3
"""Record reviewed PNGs from the interrupted job without re-reading dataset arrays.

This bounded recovery is specific to job 23941309. Dates were read from the plot
headings; selection and crop rules come from its archived exporter. Missing
array hashes, crop offsets and confusion counts remain explicitly unavailable.
"""
import hashlib
import json
from pathlib import Path
import argparse


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--export',type=Path,required=True)
    a=parser.parse_args()
    names={'timeline':'Observed fire changes across dates',
           'fire-mask':'A numeric active-fire channel becomes a binary mask',
           'channels':'Different features describe the same patch',
           'missingness':'Gray means deliberately hidden, not observed background',
           'persistence':"Yesterday's mask is a simple forecast to challenge"}
    manifest={'schema':1,'job_id':'23941309','job_state':'FAILED after completing two figure sets; third event had no qualifying window',
       'source_commit':'8ca0932ab8717e70b8e131c80bfc5ec47ce88358',
       'source_dataset':'WSTS original 2021 events in the corrected project HDF5 collection',
       'source_record':'https://zenodo.org/records/8006177',
       'selection':'Three fixed inventory records; first consecutive three-date window with positive last-input and target masks, target indices 6..19. Two records yielded images. No rerun or expanded search.',
       'spatial_scope':'128x128 source-pixel center crop; nearest-neighbor display; not georeferenced or representative of independent fire events.',
       'model_scope':'No neural model inference. Persistence copies the preceding observed binary mask.',
       'recovery':'PNG content inspected individually. Dates transcribed from timeline headings and verified against the other labelled plots. Original array metadata was not saved before the interruption; numerical counts and array checksums cannot be recovered from these images.',
       'skipped_events':[{'source_file':'2021/fire_24935885.hdf5','reason':'No qualifying window within fixed scan bound'}],
       'events':[]}
    for ident in ('fire_24935867','fire_24935874'):
        event={'id':ident,'source_file':'2021/'+ident+'.hdf5','dates':['2021-02-15','2021-02-16','2021-02-17'],
           'crop_rule':'Center 128x128; source offsets not retained',
           'persistence_counts':None,'array_digests':None,
           'feature_display':'Stored raw values, independent per-channel 2nd/98th percentile color limits. No unit conversion.', 'figures':[]}
        for name,title in names.items():
            path=ident+'/'+name+'.png';data=(a.export/path).read_bytes()
            if data[:8]!=b'\x89PNG\r\n\x1a\n':raise ValueError(path)
            event['figures'].append({'name':name,'title':title,'path':path,'sha256':hashlib.sha256(data).hexdigest(),
                'width':int.from_bytes(data[16:20],'big'),'height':int.from_bytes(data[20:24],'big')})
        manifest['events'].append(event)
    target=a.export/'manifest.json'
    if target.exists():raise ValueError('Refusing to replace a manifest')
    target.write_text(json.dumps(manifest,indent=2)+'\n')
    print('Recorded provenance for 10 individually reviewed PNGs; unavailable array metadata remains null.')

if __name__=='__main__':main()
