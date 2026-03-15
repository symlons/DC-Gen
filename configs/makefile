KCL = kcl run
OUT = kcl run

all: config_2d_local.yaml config_3d_local.yaml config_2d_cluster.yaml config_3d_cluster.yaml

config_2d_local.yaml:   base.k local.k 2d.k
	$(KCL) $^ -o $@

config_3d_local.yaml:   base.k local.k 3d.k
	$(KCL) $^ -o $@

config_2d_cluster.yaml: base.k cluster.k 2d.k
	$(KCL) $^ -o $@

config_3d_cluster.yaml: base.k cluster.k 3d.k
	$(KCL) $^ -o $@

clean:
	rm -f *.yaml
