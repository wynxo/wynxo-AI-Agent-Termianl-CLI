# WYNXO project instructions

## Terminal application commands

When the user asks to open a terminal application and then type, run, or execute a command in it, treat that as one request and use `launch_application` with BOTH arguments:

- `query`: the terminal application name, such as `konsole`
- `command`: the exact command the user asked to run, such as `echo hello`

Do not call `launch_application` with only `query` when the user explicitly asks for a command to be run in the opened terminal. The terminal launcher already keeps the terminal open after the command finishes.

Examples:
- "open konsole and write `echo hello`" → `launch_application(query="konsole", command="echo hello")`
- "open konsole and run `pwd`" → `launch_application(query="konsole", command="pwd")`
- "open konsole" → `launch_application(query="konsole")`
