from abaqus import *
from abaqusConstants import *
from caeModules import *
import os
import time
import math

def BatchAnalysis(L1, h1, R1, R2, i1, T1, T2, i2, r1, r2, i3, t1, t2, i4, a1, a2, i5, P1, P2, i6, Density, Elastic1, Elastic2, Plastic1, Plastic2, Z_N, X_N, Y_N, Z_NM, X_NM, Y_NM, CPU, GPU):
    Date = (time.strftime('%Y%m%d', time.localtime(time.time())))
    T_values = [random.uniform(T1, T2) for _ in range(i1)]
    t_values = [random.uniform(t1, t2) for _ in range(i2)]
    R_values = [random.uniform(R1, R2) for _ in range(i3)]
    r_values = [random.uniform(r1, r2) for _ in range(i4)]
    a_values = [random.uniform(a1, a2) for _ in range(i5)]
    P_values = [random.uniform(P1, P2) for _ in range(i6)]
    for T in T_values:
        for t in t_values:
            for R in R_values:
                for r in r_values:
                    for a in a_values:
                        for P in P_values:
                            L = L1
                            h =h1
                            H = h + R + T
                            A1 = (0, math.cos(math.radians(a))*H + math.sin(math.radians(a))*(r+t), 0.5*L - math.sin(math.radians(a)) * H + math.cos(math.radians(a)) * (r+t))
                            A2 = (0, math.cos(math.radians(a))*H + math.sin(math.radians(a))*(r), 0.5*L - math.sin(math.radians(a)) * H + math.cos(math.radians(a)) * (r))
                            B = (0, math.cos(math.radians(a))*H, 0.5*L - math.sin(math.radians(a))*H)
                            C1 = (0, math.cos(math.radians(a))*H - math.sin(math.radians(a))*(r+t), 0.5*L - math.sin(math.radians(a)) * H - math.cos(math.radians(a)) * (r+t))
                            C2 = (0, math.cos(math.radians(a))*H - math.sin(math.radians(a))*(r), 0.5*L - math.sin(math.radians(a)) * H - math.cos(math.radians(a)) * (r))
                            MaterialName = 'MaterialName1'
                            CyDensity = Density
                            CyElastic = [Elastic1, Elastic2]
                            CYPlastic = [Plastic1, Plastic2]
                            meshSize = int(20-0.2*a)
                            layerNum = int(3+0.06*a)
                            CYCf = [Z_N, X_N, Y_N]
                            CYCm = [Z_NM, X_NM, Y_NM]
                            CYPressure = P
                            jobName = 'Local_pipe_analysis_Job'
                            annotation = str(Date) + '_' + str(R) + '-' + str(T) + '-' + str(r) + '-' + str(t) + 'mm_' + str(
                                a) + "d_" + str(CYPressure) + 'MPa'
                            CpuNumber = CPU
                            GpuNumber = GPU
                            fail_path = os.path.join("C:\\temp\\Local_pipe_analysis\\")
                            dir_path = (fail_path + str(Date) + '-' + str(R) + '-' + str(
                                T) + '-' + str(r) + '-' + str(t) + 'mm_' + str(a) + "d_" + str(CYPressure) + 'MPa')
                            os.makedirs(dir_path)
                            os.chdir(dir_path)
                            cliCommand("""Mdb()""")
                            mod = mdb.models["Model-1"]
                            skt = mod.ConstrainedSketch(name='__profile__', sheetSize=2000.0)
                            skt.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(R, 0.0))
                            skt.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(R+T, 0.0))
                            pObj1 = mod.Part(name='tongti-1', dimensionality=THREE_D, type=DEFORMABLE_BODY)
                            pObj1.BaseSolidExtrude(sketch=skt, depth=L)
                            del mod.sketches['__profile__']
                            skt = mod.ConstrainedSketch(name='__profile__', sheetSize=2000.0)
                            skt.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(r, 0.0))
                            skt.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(r+t, 0.0))
                            pObj2 = mod.Part(name='jieguan-1', dimensionality=THREE_D, type=DEFORMABLE_BODY)
                            pObj2.BaseSolidExtrude(sketch=skt, depth=H)
                            del mod.sketches['__profile__']
                            skt = mod.ConstrainedSketch(name='__profile__', sheetSize=2000.0)
                            skt.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(R+T, 0.0))
                            pObj3 = mod.Part(name='yuanzhu-1', dimensionality=THREE_D, type=DEFORMABLE_BODY)
                            pObj3.BaseSolidExtrude(sketch=skt, depth=L)
                            del mod.sketches['__profile__']
                            skt = mod.ConstrainedSketch(name='__profile__', sheetSize=2000.0)
                            skt.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(r, 0.0))
                            pObj4 = mod.Part(name='yuanzhu-2', dimensionality=THREE_D, type=DEFORMABLE_BODY)
                            pObj4.BaseSolidExtrude(sketch=skt, depth=H)
                            del mod.sketches['__profile__']
                            asm = mod.rootAssembly
                            asm.DatumCsysByDefault(CARTESIAN)
                            asm.Instance(name='tongti-1-1', part=pObj1, dependent=ON)
                            asm.Instance(name='yuanzhu-2-1', part=pObj4, dependent=ON)
                            asm = mod.rootAssembly
                            asm.rotate(instanceList=('yuanzhu-2-1',), axisPoint=(0.0, 0.0, 0.0),
                                       axisDirection=(10.0, 0.0, 0.0), angle=-90-a)
                            asm.translate(instanceList=('yuanzhu-2-1',), vector=(0.0, 0.0, 0.5 * L))
                            asm.InstanceFromBooleanCut(name='tongti',
                                                       instanceToBeCut=mod.rootAssembly.instances['tongti-1-1'],
                                                       cuttingInstances=(asm.instances['yuanzhu-2-1'],),
                                                       originalInstances=DELETE)
                            asm.Instance(name='jieguan-1-1', part=pObj2, dependent=ON)
                            asm.Instance(name='yuanzhu-1-1', part=pObj3, dependent=ON)
                            asm = mod.rootAssembly
                            asm.rotate(instanceList=('jieguan-1-1',), axisPoint=(0.0, 0.0, 0.0),
                                       axisDirection=(10.0, 0.0, 0.0), angle=-90-a)
                            asm.translate(instanceList=('jieguan-1-1',), vector=(0.0, 0.0, 0.5 * L))
                            asm.InstanceFromBooleanCut(name='jieguan',
                                                       instanceToBeCut=mod.rootAssembly.instances['jieguan-1-1'],
                                                       cuttingInstances=(asm.instances['yuanzhu-1-1'],),
                                                       originalInstances=DELETE)
                            asm.InstanceFromBooleanMerge(name='zhengti', instances=(asm.instances['tongti-1'],
                                                                                    asm.instances['jieguan-1'],), keepIntersections=ON,
                                                         originalInstances=DELETE, domain=GEOMETRY)
                            mod.Material(name=MaterialName)
                            mod.materials[MaterialName].Density(table=((CyDensity,),))
                            mod.materials[MaterialName].Elastic(table=((CyElastic[0], CyElastic[1]),))
                            mod.materials[MaterialName].Plastic(table=((CYPlastic[0], CYPlastic[1]),))
                            mod.HomogeneousSolidSection(name='Section-1', material=MaterialName, thickness=None)
                            pObj = mod.parts['zhengti']
                            pickedFaces1 = pObj.faces.getByBoundingCylinder([0, 0, -1E5], [0, 0, 1E5], 0.5 * sum([R, R+T]))
                            surf1 = pObj.Surface(name="Surf-InFaces1", side1Faces=pickedFaces1)
                            pickedFaces2 = pObj.faces.getByBoundingCylinder([0, 0, 0.5 * L], [0, math.cos(math.radians(a))*(H + 1E-3),
                                                                                              0.5 * L - math.sin(math.radians(a))*(H + 1E-3)], 0.5 * sum([r, r+t]))
                            surf2 = pObj.Surface(name="Surf-InFaces2", side1Faces=pickedFaces2)
                            surfIn = pObj.SurfaceByBoolean(name="Surf-InFaces", surfaces=[surf1, surf2], operation=UNION)
                            pObj.Set(name="Set-InFaces", faces=surfIn.faces)
                            del pObj.surfaces["Surf-InFaces1"]
                            del pObj.surfaces["Surf-InFaces2"]
                            surf1 = pObj.Surface(name="Surf-AllFaces", side1Faces=pObj.faces)
                            pickedFaces = pObj.faces.getByBoundingCylinder([0, 0, -1E-3], [0, 0, 1E-3], R+T + 1E-3)
                            pickedEdges = pObj.edges.getByBoundingBox(-1E5, -1E5, -1.0, 1E5, 1E5, 1.0)
                            surf2 = pObj.Surface(name="Surf-Tmp1", side1Faces=pickedFaces)
                            set1 = pObj.Set(name="Set-Tmp1", edges=pickedEdges, faces=pickedFaces)
                            pickedFaces = pObj.faces.getByBoundingCylinder([0, 0, L - 1E-3], [0, 0, L + 1E-3],
                                                                           R+T + 1E-3)
                            pickedEdges = pObj.edges.getByBoundingBox(-1E5, -1E5, L - 1.0, 1E5, 1E5, L + 1.0)
                            surf3 = pObj.Surface(name="Surf-Tmp2", side1Faces=pickedFaces)
                            set2 = pObj.Set(name="Set-Tmp2", edges=pickedEdges, faces=pickedFaces)
                            pickedFaces = pObj.faces.getByBoundingCylinder([0, math.cos(math.radians(a))*(H + 10), 0.5*L - math.sin(math.radians(a))*(H + 10)],
                                                                           [0, math.cos(math.radians(a))*(H - 1E-3), 0.5*L - math.sin(math.radians(a))*(H - 1E-3)], r+t + 1E-3)
                            surf4 = pObj.Surface(name="Surf-Tmp3", side1Faces=pickedFaces)
                            pickedFaces = pObj.faces.getByBoundingCylinder([0, math.cos(math.radians(a))*(H - 1E-3), 0.5*L - math.sin(math.radians(a))*(H - 1E-3)],
                                                                           [0, 0, 0.5*L], r+t + 1E-3)
                            surf5 = pObj.Surface(name="Surf-Tmp4", side1Faces=pickedFaces)
                            surf6 = pObj.SurfaceByBoolean(name="Surf-ExFaces", surfaces=[surf1, surf2, surf3, surf4, surf5, surfIn],
                                                          operation=DIFFERENCE)
                            pObj.Set(name="Set-ExFaces", faces=surf6.faces)
                            pObj.SetByBoolean(name="Set-Fixed", sets=[set1, set2], operation=UNION)
                            del pObj.surfaces["Surf-Tmp1"]
                            del pObj.surfaces["Surf-Tmp2"]
                            del pObj.surfaces["Surf-Tmp3"]
                            del pObj.surfaces["Surf-Tmp4"]
                            del pObj.sets["Set-Tmp1"]
                            del pObj.sets["Set-Tmp2"]
                            region = regionToolset.Region(cells=pObj.cells)
                            pObj.SectionAssignment(region=region, sectionName='Section-1', offset=0.0,
                                                   offsetType=MIDDLE_SURFACE, offsetField='',
                                                   thicknessAssignment=FROM_SECTION)
                            pObj.PartitionCellByPlaneThreePoints(cells=pObj.cells, point1=[0, 0, 0], point2=[1, 0, 0], point3=[0, 0, 1])
                            pickedCells = pObj.cells.getByBoundingBox(-1E5, -1E-3, -1E5, 1E5, 1E5, 1E5)
                            faces = pObj.faces
                            for f in faces:
                                try:
                                    if abs(f.getRadius() - r-t) < 1E-3:
                                        pObj.PartitionCellByExtendFace(extendFace=f, cells=pickedCells)
                                        break
                                except:
                                    pass
                            pickedCells = pObj.cells.getByBoundingBox(-1E5, -1E-3, -1E5, 1E5, 1E5, 1E5)
                            pObj.PartitionCellByPlaneThreePoints(cells=pickedCells, point1=[0, 0, 0], point2=[0, 1, 0], point3=[0, 0, 1])
                            if a < 30:
                                pickedCells = pObj.cells.getByBoundingBox(-1E5, -1E-3, -1E5, 1E5, 1E5, 1E5)
                                pObj.PartitionCellByPlaneNormalToEdge(edge=pObj.edges[26], point=pObj.vertices[19], cells=pickedCells)
                            else:
                                pass
                            pObj.seedPart(size=meshSize, deviationFactor=0.1, minSizeFactor=0.1)
                            pickedEdges = pObj.edges.getByBoundingBox(-1E5, -1E-3, -1E-3, 1E5, 1E-3, 1E-3)
                            pObj.seedEdgeByNumber(edges=pickedEdges, number=layerNum, constraint=FINER)
                            pickedEdges = pObj.edges.getByBoundingBox(-1E-5, C1[1] - 1E-3, 1E-3,
                                                                      1E-5, A1[1] + 1E-3, L - 1E-3)
                            pObj.seedEdgeByNumber(edges=pickedEdges, number=layerNum, constraint=FINER)
                            pObj.generateMesh()
                            mod.StaticStep(name='Step-1', previous='Initial', maxNumInc=10000, minInc=1e-10)
                            asm = mod.rootAssembly
                            asm.ReferencePoint(point=[B[0], B[1], B[2]])
                            asm.regenerate()
                            refPoints1 = asm.referencePoints.values()
                            region1 = regionToolset.Region(referencePoints=refPoints1)
                            insObj = asm.instances['zhengti-1']
                            side1Faces1 = insObj.faces.getByBoundingBox(-1E5, C1[1] - 1E-5, -1E5, 1E5, A1[1] + 1E-5, 1E5)
                            region2 = regionToolset.Region(side1Faces=side1Faces1)
                            mod.Coupling(name='Constraint-1', controlPoint=region1,
                                         surface=region2, influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC,
                                         localCsys=None, u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)
                            region = regionToolset.Region(referencePoints=refPoints1)
                            mod.ConcentratedForce(name='Load-1', createStepName='Step-1',
                                                  region=region, cf1=CYCf[0], cf2=CYCf[1], cf3=CYCf[2],
                                                  distributionType=UNIFORM,
                                                  field='', localCsys=None)
                            region = regionToolset.Region(referencePoints=refPoints1)
                            mod.Moment(name='Load-2', createStepName='Step-1',
                                       region=region, cm1=CYCm[0], cm2=CYCm[1], cm3=CYCm[2], distributionType=UNIFORM,
                                       field='', localCsys=None)
                            region = insObj.surfaces['Surf-InFaces']
                            mod.Pressure(name='Load-3', createStepName='Step-1',
                                         region=region, distributionType=UNIFORM, field='', magnitude=CYPressure,
                                         amplitude=UNSET)
                            region = insObj.sets['Set-Fixed']
                            mod.EncastreBC(name='BC-1', createStepName='Step-1', region=region, localCsys=None)
                            mdb.Job(name=jobName, model='Model-1', numCpus=CpuNumber, description=annotation, numDomains=CpuNumber, numGPUs=GpuNumber)
                            mdb.jobs[jobName].submit(consistencyChecking=OFF)
                            mdb.jobs[jobName].waitForCompletion()
                            odbObj = session.openOdb(name=jobName + '.odb')
                            viewport = session.viewports['Viewport: 1']
                            viewport.setValues(displayedObject=odbObj)
                            viewport.odbDisplay.setPrimaryVariable(variableLabel='S', outputPosition=INTEGRATION_POINT,
                                                                   refinement=(INVARIANT, 'Mises'), )
                            viewport.odbDisplay.display.setValues(plotState=(CONTOURS_ON_UNDEF,))
                            maxMisesNode = int(viewport.getPrimVarMinMaxLoc()['maxPosition'].split(":")[-1])
                            insObj = odbObj.rootAssembly.instances.values()[-1]
                            nodes = insObj.nodes
                            node2coord = {}
                            for n in nodes:
                                node2coord[n.label] = n.coordinates
                            disp = odbObj.steps.values()[-1].frames[-1].fieldOutputs["U"].values
                            for p in disp:
                                nid = p.nodeLabel
                                node2coord[nid] = [node2coord[nid][i] + p.data[i] for i in range(3)]
                            maxMisesNodeCoord = node2coord[maxMisesNode]
                            exFaceNodes = insObj.nodeSets['SET-EXFACES'].nodes
                            minDist = 1E15
                            minDistNid = -1
                            for n in exFaceNodes:
                                nid = n.label
                                p1 = node2coord[nid]
                                dist = sqrt(
                                    (maxMisesNodeCoord[0] - p1[0]) ** 2 + (maxMisesNodeCoord[1] - p1[1]) ** 2 + (maxMisesNodeCoord[2] - p1[2]) ** 2)
                                if dist < minDist:
                                    minDist, minDistNid = dist, nid
                            print("Max Mises Node:,", maxMisesNode, "; Min Distance Node:", minDistNid)
                            session.Path(name='Path-MaxMisesNode2ExFace', type=NODE_LIST,
                                         expression=((insObj.name, (maxMisesNode, minDistNid,)),))
                            minDistNodeCoord = node2coord[minDistNid]
                            line1 = [minDistNodeCoord, [0.0, minDistNodeCoord[1], 0.5 * L]]
                            line2 = [minDistNodeCoord, [0.0, 0.0, minDistNodeCoord[2]]]
                            v1, v2 = [line1[1][i] - line1[0][i] for i in range(3)], [line2[1][i] - line2[0][i] for i in range(3)]
                            d1, d2 = sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2), sqrt(v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2)
                            v1, v2 = [v1[i] / d1 for i in range(3)], [v2[i] / d2 for i in range(3)]
                            inFaceNodes = insObj.nodeSets['SET-INFACES'].nodes
                            minDist1 = [1E15, 1E15]
                            minDistNid1 = [-1, -1]
                            for n in inFaceNodes:
                                nid = n.label
                                p1 = node2coord[nid]
                                projDist1 = (p1[0] - line1[0][0]) * v1[0] + (p1[1] - line1[0][1]) * v1[1] + (p1[2] - line1[0][2]) * v1[2]
                                if 0 <= projDist1 <= 0.2 * d1:
                                    dist = (p1[0] - line1[0][0]) ** 2 + (p1[1] - line1[0][1]) ** 2 + (p1[2] - line1[0][2]) ** 2
                                    dist = sqrt(dist - projDist1 ** 2)
                                    if dist < minDist1[0]:
                                        minDist1[0], minDistNid1[0] = dist, nid
                                projDist2 = (p1[0] - line2[0][0]) * v2[0] + (p1[1] - line2[0][1]) * v2[1] + (p1[2] - line2[0][2]) * v2[2]
                                if 0 <= projDist2 <= 0.1 * d2:
                                    dist = (p1[0] - line2[0][0]) ** 2 + (p1[1] - line2[0][1]) ** 2 + (p1[2] - line2[0][2]) ** 2
                                    dist = sqrt(dist - projDist2 ** 2)
                                    if dist < minDist1[1]:
                                        minDist1[1], minDistNid1[1] = dist, nid
                            print("Node:,", minDistNid, "; Min Distance Node:", minDistNid1)
                            session.Path(name='Path-MinDistNode2InFace1', type=NODE_LIST,
                                         expression=((insObj.name, (minDistNid, minDistNid1[0],)),))
                            session.Path(name='Path-MinDistNode2InFace2', type=NODE_LIST,
                                         expression=((insObj.name, (minDistNid, minDistNid1[1],)),))
                            path = session.paths['Path-MaxMisesNode2ExFace']
                            xyList = session.linearizeStress(name='SCL-AA', path=path, modelShape=DEFORMED,
                                                             xyMembraneComps=('S11', 'S22', 'S33',),
                                                             xyBendingComps=('S11', 'S22', 'S33',),
                                                             curvatureCorrection=False, saveXy=True, writeReport=True,
                                                             reportFile='linearStressSCL-AA.rpt', )
                            xyp = session.XYPlot('XYPlot-1')
                            chartName = xyp.charts.keys()[0]
                            chart = xyp.charts[chartName]
                            curveList = session.curveSet(xyList)
                            chart.setValues(curvesToPlot=curveList)
                            session.charts[chartName].autoColor(lines=True, symbols=True)
                            session.viewports['Viewport: 1'].setValues(displayedObject=xyp)
                            path = session.paths['Path-MinDistNode2InFace1']
                            xyList = session.linearizeStress(name='SCL-BB', path=path, modelShape=DEFORMED,
                                                             xyMembraneComps=('S11', 'S22', 'S33',),
                                                             xyBendingComps=('S11', 'S22', 'S33',),
                                                             curvatureCorrection=False, saveXy=True, writeReport=True,
                                                             reportFile='linearStressSCL-BB.rpt', )
                            xyp = session.xyPlots['XYPlot-1']
                            chartName = xyp.charts.keys()[0]
                            chart = xyp.charts[chartName]
                            curveList = session.curveSet(xyList)
                            chart.setValues(curvesToPlot=curveList)
                            session.charts[chartName].autoColor(lines=True, symbols=True)
                            path = session.paths['Path-MinDistNode2InFace2']
                            xyList = session.linearizeStress(name='SCL-CC', path=path, modelShape=DEFORMED,
                                                             xyMembraneComps=('S11', 'S22', 'S33',),
                                                             xyBendingComps=('S11', 'S22', 'S33',),
                                                             curvatureCorrection=False, saveXy=True, writeReport=True,
                                                             reportFile='linearStressSCL-CC.rpt', )
                            xyp = session.xyPlots['XYPlot-1']
                            chartName = xyp.charts.keys()[0]
                            chart = xyp.charts[chartName]
                            curveList = session.curveSet(xyList)
                            chart.setValues(curvesToPlot=curveList)
                            session.charts[chartName].autoColor(lines=True, symbols=True)
                            mdb.saveAs(pathName=dir_path + '/Local_pipe_analysis')
                            del session.xyPlots['XYPlot-1']
                            odbObj = session.openOdb(name=jobName + '.odb')
                            step_name = 'Step-1'
                            frame_num = -1
                            output_field = 'S'
                            step = odbObj.steps[step_name]
                            last_frame = step.frames[frame_num]
                            field = last_frame.fieldOutputs[output_field]
                            max_mises = 0.0
                            for i in range(len(field.values)):
                                if field.values[i].mises > max_mises:
                                    max_mises = field.values[i].mises
                            full_path = os.path.join(fail_path, 'MaxMises.txt')
                            f = open(full_path, 'a+')
                            f.write('MaxMises,' + annotation + ',' + str(max_mises) + '\n')
                            f.close()
