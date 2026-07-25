- when looking for machines, you use vastai or verda depending on what the user asks
- make a table when looking for machine that informs user (location, type, cost)
- when renting on Vast.ai, use the recommended `PyTorch (Vast)` template instead of launching directly from a CUDA image, unless the user asks otherwise
- make the changes locally - but ask before commiting or pushing



## Launching machines: 
- You have access to vastai cli 
- I like machines that are like H100, high internet speeds, trusted datacenter - preferably in europe 
- Confirm the last line of the run.sh script is the one we want to run (no need to read the entire file) 
- Confirm batch size, etc, and nproc detection is ok in the run we want to run
- copy run.sh into the remote machine when rented with scp only. 
- do not inspect commit state, copy local diffs, or alter its cloning workflow unless the script fails or the user asks
- the run.sh will trigger training in one tmux window, and `uvx nvitop` in another window of the same session
