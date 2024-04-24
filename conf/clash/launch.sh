#!/bin/bash
gnome-terminal --tab --title=clash -- bash -c "gsettings set org.gnome.system.proxy mode 'manual' && cd /home/jzi/clash && ./clash -d .;exec bash" 
gnome-terminal --tab --title=edge -- bash -c "microsoft-edge https://clash.razord.top/"
gnome-terminal --tab --title=proxyclose -- bash -c "/home/jzi/clash/proxyclose && gsettings set org.gnome.system.proxy mode 'none' && exit;exec bash"
