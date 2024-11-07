.. Modified Files
================

This folder contains the ns-3 files that have been modified in this project.
Please replace these files with the modified ones before running.

.. toctree::
   :maxdepth: 2

Modified Files
---------------

The following files have been modified:

* :file:`frame-exchange-manager.h` - Make FrameExchangeManager::Receive virtual
* :file:`sta-wifi-mac.h` - Make some private methods protected
* :file:`file3.cpp` - Brief description of changes made to file3.cpp

.. note::
   This list only includes files that have been modified, and does not include new files or deleted files.

File Details
------------

### frame-exchange-manager.h

* :ref:`frame-exchange-manager.h <src/wifi/model/frame-exchange-manager.h>` 
   - Modify FrameExchangeManager to make the Receive method virtual. This change enables the PanlinkFrameExchangeManager to override the Receive method, allowing it to collect and process all received frames, regardless of whether they are addressed to the current device.

### sta-wifi-mac.h

* :ref:`sta-wifi-mac.h <src/wifi/model/sta-wifi-mac.h>` 
   - Make two private methods DoDispose and DoIntialize protected, so that they can be override in the derived classes.

### file3.cpp

* :ref:`file3.cpp <file3_cpp>` 
   - Detailed description of changes made to file3.cpp