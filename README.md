# qvarnet: A package to simulate quantum systems via VMC with Artificial Neural Networks ansatzë

### Installation

To install the conda environment, from the root directory of the project run:

```bash
conda env create -f environment_config.yaml
```

and activate with

```bash
conda activate jax
```

Finally, from the root directory of the project, the package must be installed in edit mode:

```bash
pip install -e .
```

### Execution

From the root folder, run

```bash
qvarnet run [>output.txt]
```

The parameters are found inside `./src/qvarnet/cli/parameters`

The [] option is recommended if there are prints in the code. The progressbar is still displayed in the terminal

# WIP: Diagrams

## Class Diagram

```mermaid
classDiagram
    %% Relations
    Animal <|-- Dog : Inherits
    Animal <|-- Cat : Inherits
    Owner "1" o-- "1..*" Animal : Aggregation (Has-a)

    %% Base Class
    class Animal {
        +String name
        +int age
        #String _chip_id
        +make_sound() str
        +eat(food: str) void
    }

    %% Subclass 1
    class Dog {
        +String breed
        +bark() void
        -chase_tail() void
    }

    %% Subclass 2
    class Cat {
        +bool loves_catnip
        +meow() void
    }
    
    %% Related Class
    class Owner {
        +String name
        +List~Animal~ pets
        +adopt_pet(new_pet: Animal)
    }
```

Flowchart Diagram

```mermaid
flowchart TD
    %% Nodes
    A([Start])
    B[/Read Input File/]
    C{Is Data Clean?}
    D[Process Data]
    E[Log Error]
    F[(Save to DB)]
    G([End])

    %% Edge Connections
    A --> B
    B --> C
    C -- Yes --> D
    C -- No --> E
    E --> G
    D --> F
    F --> G
```