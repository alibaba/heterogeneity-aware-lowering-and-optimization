#!/bin/bash
git config clangformat.binary clang-format-9

# ID for empty tree object.
against="4b825dc642cb6eb9a060e54bf8d69288fbee4904"
if git rev-parse --verify HEAD >/dev/null 2>&1 ; then
  against="HEAD"
fi

tmp=`mktemp`
rc=0
git clang-format-9 $against > $tmp
first_line=`head -n1 $tmp`
if [[ $first_line =~ 'changed files:' ]]; then
  rc=1
  echo "Commit aborted. Please run 'git diff' to review the changes and re-commit again."
  cat $tmp
  echo "Run 'git clang-format-9 -f origin/master' to re-verify the format"
fi
rm $tmp
exit $rc