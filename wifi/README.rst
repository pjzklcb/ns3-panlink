.. Modified Files
================

This folder contains the ns-3 files that have been modified in this project.
Please replace these files with the modified ones before running.

.. toctree::
   :maxdepth: 2

Modified Files
---------------

The following files have been modified:

* :file:`frame-exchange-manager.h` - Make some methods virtual
* :file:`sta-wifi-mac.h` - Make some private methods protected
* :file:`txop.h` - Make some methods virtual
* :file:`channel-access-manager.h` - Make some methods protected or virtual

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

### txop.h

* :ref:`txop.h <src/wifi/model/txop.h>` 
   - Make Txop::UpdateBackoffSlotsNow virtual.

### channel-access-manager.h

* :ref:`channel-access-manager.h <src/wifi/model/channel-access-manager.h>`
   - Make all of the private members and methods protected.
   - Make methods related to access defer virtual, so that restore access can be scheduled.