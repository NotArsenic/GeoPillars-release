To pass our data to our backbone, we first prepare features from the pillarized point clouds.
Pillar Feature Builder transforms a set of pillars into rich feature vectors.
Given a set of `M` pillars, it computes:
1. Intra-pillar cluster offsets (f_cluster)
2. BEV pillar-center offsets (f_center)
3. Euclidian range (d)
4. Azimuth angle (theta)
5. Pillar-level point density scaler (p)
These additional features are combined with the raw input features (xyz, Intensity) to produce a C-dimensional encoding per point. Invalid (padded) points are supressed via a validity mask derived from per-pillar point counts.