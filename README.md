# Trunks

Trunks helps you manage a set of ephemeral branches for merging features you're working on using a stacked-diffs approach.

Imagine you are working on several small tweaks and a few bigger related and possibly dependent change-sets. Your commit history looks like this:

```
       main --> c5 Support new workload type
                c4 Improve work distribution algorithm
                c3 Add max memory property to Worker class
                c2 Increase worker threads
origin/main --> c1 Fix race condition in scheduler
```

You have a few *local commits* to `main`, your *local trunk*, which is ahead of `origin/main`, your *remote trunk*.

Two commits, `c2` and `c3` are self-contained changesets that can be merged independently, but `c4` depends on `c2` and `c5` depends on `c4`. In order words, we could create two pull requests into `main` containing commits `c2` and `c3` respectively, one pull request containing `c4` into the branch containing `c2` and one pull request with `c5` into the branch containing `c4`.

## Creating a plan

Trunks lets you enter this information into a *plan* based on which it creates a tree of ephemeral branches off `main` that support the pull requests described above.

Invoking `trunk edit` at this point will open an editor showing your local commits and the current plan:

```
s c2 Increase worker threads
s c3 Add max memory property to Worker class
s c4 Improve work distribution algorithm
s c5 Support new workload type
```

This default plan will not do anything because it instructs trunks to skip (`s`) every commit.

To instruct trunks to create the branching structure described above we use the `b` instruction coupled with an optional numeric label and target branch:

```
b     c2 Increase worker threads
b1    c3 Add max memory property to Worker class
b2@b1 c4 Improve work distribution algorithm
b3@b2 c5 Support new workload type
```

This will create four branches with the following dependencies:

```
main <---- b <-- b2 <-- b3
       `-- b1
```

## Updating the plan

After submitting your pull requests for review you will likely have to make a few amendments to your work. It is not necessary to switch branches in order to make these changes. You can work on your local trunk and freely use git (interactive) rebases to incorporate your changes into existing commits.

Additionally, you may want to fetch main and rebase your local commits on top of the work of others.

Imagine that after updating your local work and fetching and rebasing your local branch, it looks like this:

```
       main --> c7 Support new workload type
                c6 Improve workload distribution algorithm
                c5 Add max memory property to Worker class
                c4 Address reviewer comments                 # new local commit
                c3 Increase worker threads
origin/main --> c2 Increase scheduler performance            # updated remote main branch
                c1 Fix race condition in scheduler
```

Since you have rebased, you will need to update your pull requests. Doing this manually with git would require going into each branch, rebasing it against its target branch and replacing the commits you want to have reviewed (e.g., with a combination of `git reset --hard` and `git cherry-pick`).

Trunks automates this. Run `trunks update` to delete and recreate your local ephemeral branches based on the plan.

Trunks remembers the plan, even though due to rebasing, the commit SHAs may have changed. Trunks uses the commit message of the last commit in a branch to match branches to your local commits. As long as these don't change, trunks will recognize them and be able to reconstruct the plan.

This will update (in fact, delete and recreate) the ephemeral branches, but will not include the new commit, `c4` in the merge request for `c3`. To do that, we can edit the plan again with `trunks edit`. It now show up as follows:

```
b     c3 Increase worker threads
s     c4 Address reviewer comments
b1    c5 Add max memory property to Worker class
b2@b1 c6 Improve work distribution algorithm
b3@b2 c7 Support new workload type
```

To include `c6` in `b`, simply change its `s` into `b`:

```
b     c3 Increase worker threads
b     c4 Address reviewer comments
b1    c5 Add max memory property to Worker class
b2@b1 c6 Improve work distribution algorithm
b3@b2 c7 Support new workload type
```

After saving and exiting, trunks will re-create the local ephemeral branches based on the amended plan.

# Workflow example

Updating local commits:

`git fetch`
`git rebase -i origin/develop`

Updating 

# Epilogue

Trunks is meant to be **minimal** and only support actions that would be tedious to do manually in `git`. It attempts to **follow established idioms** in git such as editing instructions for a sequence of actions to be performed in a plain-text file in the same way interactive rebases work in git. Finally, trunks is **stateless** in the sense that trunks stores all information it needs to persist in git branch names.

The branches it creates are called ephemeral because will never need to (and perhaps never should) work directly on any of them. Instead, you can do all your work and make amendments only in your local trunk.

This tool is intended for developers who don't need to be told how to re-order and slice and dice commits in their local history with git.

