# # Dry run first — see every match without changing anything
# grep -rn '\.at({' tests/unit/loss_functions

# # Then do the replacement
# find src -name "*.cpp" -o -name "*.h" | xargs sed -i 's/\.at({\([^}]*\)})/\.at(\1)/g'

find src tests -name "*.cpp" -o -name "*.h" | xargs sed -i 's/\.at({\([^}]*\)})/\.at(\1)/g'