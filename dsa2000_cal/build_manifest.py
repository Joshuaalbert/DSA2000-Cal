import glob

if __name__ == '__main__':
    with open('MANIFEST.in', 'w') as f:
        for path in glob.glob('**/.large_files', root_dir='dsa2000_cal', recursive=True):
            f.write(f'include {path}\n')
