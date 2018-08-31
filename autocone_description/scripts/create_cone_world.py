#!/usr/bin/env python

if __name__ == "__main__":

    num_cones = 1000
    posX = 1
    posY = 0
    posZ = 0

    init_txt = "<?xml version=\"1.0\" ?>\n"\
    "<sdf version=\"1.6\">\n"\
    "\t<world name=\"default\">\n"\
    "\t\t<scene>\n"\
    "\t\t\t<sky>\n"\
    "\t\t\t\t<clouds>\n"\
    "\t\t\t\t\t<speed>12</speed>\n"\
    "\t\t\t\t</clouds>\n"\
    "\t\t\t</sky>\n"\
    "\t\t\t<ambient>1.0 1.0 1.0 1.0</ambient>\n"\
    "\t\t\t<shadows>false</shadows>\n"\
    "\t\t</scene>\n"\
    "\t\t<!-- A ground plane -->\n"\
    "\t\t<include>\n"\
    "\t\t\t<uri>model://ground_plane</uri>\n"\
    "\t\t\t<pose>0 0 0 0 0 0</pose>\n"\
    "\t\t</include>\n\n"\
    "\t\t<include>\n"\
    "\t\t\t<uri>model://sun</uri>\n"\
    "\t\t</include>\n\n"

    end_txt = "\t</world>\n"\
    "</sdf>"

    include_init_txt = "\t\t<include>\n"
    include_end_txt = "\t\t</include>\n\n"

    txt = init_txt

    for i in range(num_cones):
        
        txt += include_init_txt
        txt += "\t\t\t<name>cone_" + str(i) + "</name>\n"
        txt += "\t\t\t<uri>model://urdf/models/mini_cone</uri>\n"
        txt += "\t\t\t<pose>" + str(posX) + " " + str(posY) + " " + str(posZ) + " 0 0 0</pose>\n"
        txt += include_end_txt

        posX += 1

        if posX%10 == 0:
            posX = 0
            posY += 1


    txt += end_txt
    
    #print(txt)

    file = open("./../urdf/cone_world.world", "w")
    file.write(txt)
    file.close()