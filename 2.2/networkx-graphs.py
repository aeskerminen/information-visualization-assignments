import networkx as nx
import matplotlib.pyplot as plt

# List of SDG names (1-17)
sdg_names = [
	"No Poverty",
	"Zero Hunger",
	"Good Health and Well-being",
	"Quality Education",
	"Gender Equality",
	"Clean Water and Sanitation",
	"Affordable and Clean Energy",
	"Decent Work and Economic Growth",
	"Industry, Innovation and Infrastructure",
	"Reduced Inequalities",
	"Sustainable Cities and Communities",
	"Responsible Consumption and Production",
	"Climate Action",
	"Life Below Water",
	"Life on Land",
	"Peace, Justice and Strong Institutions",
	"Partnerships for the Goals"
]

G = nx.Graph()

# Add one node for each SDG, with label
for i, name in enumerate(sdg_names, 1):
	G.add_node(i, label=f"SDG {i}: {name}")

G.add_edge(1, 2)
G.add_edge(1, 3)
G.add_edge(1, 4)
G.add_edge(1, 6)
G.add_edge(1, 7)
G.add_edge(1, 10)

G.add_edge(8, 1)
G.add_edge(8, 12)

G.add_edge(9, 1)
G.add_edge(9, 2)
G.add_edge(9, 7)

G.add_edge(10, 5)
G.add_edge(10, 5)

G.add_edge(11, 6)
G.add_edge(11, 15)

G.add_edge(12, 13)
G.add_edge(12, 3)

G.add_edge(13, 14)
G.add_edge(13, 15)

G.add_edge(14, 6)
G.add_edge(14, 11)

G.add_edge(16, 4)
G.add_edge(16, 5)

G.add_edge(17, 9)
G.add_edge(17, 4)
G.add_edge(17, 10)


#kamada_kawai_layout 
#spring_layout (fruchterman_reingold_layout)
#circular_layout (radial)


plt.figure(figsize=(8, 8))
pos_circ = nx.circular_layout(G)
nx.draw(
	G, pos_circ,
	with_labels=True,
	labels=nx.get_node_attributes(G, 'label'),
	node_size=2000,
	node_color='skyblue',
	font_size=8,
	font_weight='bold',
	edge_color='gray'
)
plt.title("Radial Layout")
plt.axis('off')
plt.tight_layout()
plt.savefig("sdg_radial.png", dpi=150)
plt.show()

for seed in [5, 12]:
	plt.figure(figsize=(8, 8))
	pos_kk = nx.kamada_kawai_layout(G)
	nx.draw(
		G, pos_kk,
		with_labels=True,
		labels=nx.get_node_attributes(G, 'label'),
		node_size=2000,
		node_color='lightgreen',
		font_size=8,
		font_weight='bold',
		edge_color='gray'
	)
	plt.title(f"Kamada-Kawai Layout")
	plt.axis('off')
	plt.tight_layout()
	plt.savefig(f"sdg_kamada_kawai_seed{seed}.png", dpi=150)
	plt.show()

for seed in [32, 17]:
	plt.figure(figsize=(8, 8))
	pos_fr = nx.spring_layout(G)
	nx.draw(
		G, pos_fr,
		with_labels=True,
		labels=nx.get_node_attributes(G, 'label'),
		node_size=2000,
		node_color='salmon',
		font_size=8,
		font_weight='bold',
		edge_color='gray'
	)
	plt.title(f"Fruchterman-Reingold Layout")
	plt.axis('off')
	plt.tight_layout()
	plt.savefig(f"sdg_fruchterman_seed{seed}.png", dpi=150)
	plt.show()