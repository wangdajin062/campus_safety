#!/usr/bin/env python3
"""Extract elsarticle.cls from elsarticle.dtx using the class guard."""

import re

def extract_from_dtx(dtx_path, cls_path, guards):
    """Simple docstrip implementation to extract .cls from .dtx."""
    # Normalize guards to a set
    active_guards = set(guards)

    # Stack for nested guards: list of bools
    # True = current level is "including", False = "excluding"
    # We also need to track which guard caused each level
    stack = [True]  # Start in include mode

    lines_out = []
    encountered_class_guard = False

    with open(dtx_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.rstrip('\n')

            # Check for guard lines: %<...>
            guard_match = re.match(r'^%<([*/]?)([\w,]+)>\s*(.*)$', stripped)
            if guard_match:
                prefix = guard_match.group(1)
                guard_names = set(guard_match.group(2).split(','))
                remainder = guard_match.group(3)

                if prefix == '*':
                    if 'class' in guard_names and not encountered_class_guard:
                        encountered_class_guard = True
                    should_include = guard_names.issubset(active_guards)
                    stack.append(should_include)
                elif prefix == '/':
                    if stack:
                        stack.pop()
                else:
                    # Single-line guard — output the remainder if guard active
                    should_output = guard_names.issubset(active_guards)
                    if should_output and all(stack) and remainder:
                        lines_out.append(remainder)
                continue

            # Regular line: include only if all stack levels True
            if all(stack):
                # Docstrip strips % lines (they are documentation)
                # Lines without % are code
                if stripped.startswith('%'):
                    continue
                lines_out.append(stripped)

    with open(cls_path, 'w', encoding='utf-8') as f:
        for l in lines_out:
            f.write(l + '\n')

    print(f"Extracted {len(lines_out)} lines to {cls_path}")


if __name__ == '__main__':
    extract_from_dtx('elsarticle.dtx', 'elsarticle.cls', ['class'])
