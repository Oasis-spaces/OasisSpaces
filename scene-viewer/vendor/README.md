# Vendored libraries

So the viewer works with no internet (inside the Mac and phone apps, served by
the Mac on the local network):

- `three/`: three.js 0.180.0 (MIT), the build and the few add-ons the viewer
  and Spark import (OrbitControls, GLTFLoader, BufferGeometryUtils,
  RoomEnvironment, postprocessing/Pass).
- `spark/`: Spark 2.2.0 by World Labs (MIT), https://sparkjs.dev.

To update, fetch the same files of a newer release from
https://cdn.jsdelivr.net/npm/three@<version>/ and
https://sparkjs.dev/releases/spark/<version>/spark.module.js.
