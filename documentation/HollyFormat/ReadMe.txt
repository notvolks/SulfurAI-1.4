The HollyFormat (named after my good friend) is a way to keep VH python scripts easy to read and maintain.


It consists like this:

Functions are grouped into "node classes" based on their use with hyphen-brackets used to separate them.

Hyphen-bracket: 
	#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------<TEXT>------------------------------------------

This is not only easy on the users eyes but keeps things nice and neat.

In built functions and module functions (API) should be kept separate. In built functions should start with an underscore "_".



Module functions are to be kept clean and very bug-tested with try/except trees to prevent *all* errors.

All main scripts must have their definition at the top. It should explain what the script does and whether it uses the HollyFormat (main scripts must). It is not required for secondary scripts but recommended.

All main scripts must also contain the layout of the node trees, how the scripts are laid out and what the do. It is not required for secondary scripts but recommended.
