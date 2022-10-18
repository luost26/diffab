from git import Repo


def has_changes(path='./'):
    repo = Repo(path)
    changed_files = [f.a_path for f in repo.index.diff(None)] + repo.untracked_files
    changed_files = list(filter(lambda p: not p.startswith('configs/'), changed_files))
    if len(changed_files) > 0:
        print('\n\nYou have uncommitted changes:')
        for fn in changed_files:
            print(' - %s' % fn)
        print('Please commit your changes before running the script.\n\n')
        return True
    else:
        return False


def get_version(path='./'):
    repo = Repo(path)
    return repo.active_branch.name, repo.head.object.hexsha
